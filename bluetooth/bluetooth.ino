#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <SPI.h>
#include <driver/i2s.h>
#include "ardCam.h"
#include "secrets.h"
#include <WiFiUdp.h>
#include <time.h>
#include <esp_sleep.h>


WebServer server(80);

#define SLEEP_SECONDS     3       // deep sleep duration between inferences
#define WIFI_TIMEOUT_MS   8000    // max time to wait for WiFi reconnect
#define UPLOAD_WAIT_MS    3000    // max time to wait for upload to finish before sleeping
#define NTP_RESYNC_BOOTS  30      // re-sync NTP every N boots

// ── RTC memory — survives deep sleep ─────────────────────────────────────────
RTC_DATA_ATTR int      bootCount    = 0;
RTC_DATA_ATTR time_t   rtcTime      = 0;    // last known Unix time
RTC_DATA_ATTR bool     sleepEnabled = true; // toggled from frontend

// ── Async upload queue ────────────────────────────────────────────────────────
struct UploadJob {
  uint8_t* imageBuffer;
  uint32_t imageSize;
  float    probability;
  bool     isNotAllowed;
  uint64_t timestamp;
};

static QueueHandle_t uploadQueue      = nullptr;
static TaskHandle_t  uploadTaskHandle = nullptr;

// ── Upload task (Core 0) ──────────────────────────────────────────────────────
static void uploadTask(void* param) {
  UploadJob job;
  for (;;) {
    if (xQueueReceive(uploadQueue, &job, portMAX_DELAY) != pdTRUE) continue;

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[UPLOAD] WiFi not connected, dropping job");
      free(job.imageBuffer);
      continue;
    }

    WiFiClient client;
    client.setTimeout(5000);

    if (!client.connect(LAPTOP_IP, LAPTOP_PORT)) {
      Serial.printf("[UPLOAD] ❌ Connection failed to %s:%d\n", LAPTOP_IP, LAPTOP_PORT);
      free(job.imageBuffer);
      continue;
    }

    char tsStr[25];
    sprintf(tsStr, "%llu", job.timestamp);
    String jsonData = "{";
    jsonData += "\"probability\":" + String(job.probability, 6) + ",";
    jsonData += "\"result\":\"" + String(job.isNotAllowed ? "NOT_ALLOWED" : "allowed") + "\",";
    jsonData += "\"timestamp\":" + String(tsStr);
    jsonData += "}";

    String boundary  = "----ESP32Boundary";
    String bodyStart = "--" + boundary + "\r\n";
    bodyStart += "Content-Disposition: form-data; name=\"metadata\"\r\n";
    bodyStart += "Content-Type: application/json\r\n\r\n";
    bodyStart += jsonData + "\r\n";
    bodyStart += "--" + boundary + "\r\n";
    bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"inference.raw\"\r\n";
    bodyStart += "Content-Type: application/octet-stream\r\n\r\n";
    String bodyEnd = "\r\n--" + boundary + "--\r\n";

    uint32_t totalSize = bodyStart.length() + job.imageSize + bodyEnd.length();

    client.print("POST /upload-inference HTTP/1.1\r\n");
    client.print("Host: " + String(LAPTOP_IP) + ":" + String(LAPTOP_PORT) + "\r\n");
    client.print("Content-Type: multipart/form-data; boundary=" + boundary + "\r\n");
    client.print("Content-Length: " + String(totalSize) + "\r\n\r\n");
    client.print(bodyStart);

    uint32_t sent = 0;
    while (sent < job.imageSize) {
      uint32_t toSend = min((uint32_t)CHUNK, job.imageSize - sent);
      int written = client.write(job.imageBuffer + sent, toSend);
      if (written <= 0) { Serial.println("[UPLOAD] ❌ Write failed"); break; }
      sent += written;
    }

    client.print(bodyEnd);

    unsigned long timeout = millis() + 2000;
    while (!client.available() && millis() < timeout) delay(5);
    if (client.available()) {
      Serial.printf("[UPLOAD] ✅ %s\n", client.readStringUntil('\n').c_str());
    } else {
      Serial.println("[UPLOAD] ⚠️ No response (upload likely succeeded)");
    }

    client.stop();
    free(job.imageBuffer);
  }
}

// ── sendImageToLaptopWithResult — non-blocking enqueue ───────────────────────
bool sendImageToLaptopWithResult(float probability, bool isNotAllowed) {
  extern uint8_t*  inferenceImageBuffer;
  extern uint32_t  inferenceImageSize;

  if (!inferenceImageBuffer || inferenceImageSize == 0) return false;
  if (!uploadQueue) return false;

  uint8_t* copy = (uint8_t*)malloc(inferenceImageSize);
  if (!copy) { Serial.println("[UPLOAD] malloc failed"); return false; }
  memcpy(copy, inferenceImageBuffer, inferenceImageSize);

  UploadJob job = {
    .imageBuffer  = copy,
    .imageSize    = inferenceImageSize,
    .probability  = probability,
    .isNotAllowed = isNotAllowed,
    .timestamp    = (uint64_t)time(nullptr) * 1000ULL,
  };

  if (xQueueSend(uploadQueue, &job, 0) != pdTRUE) {
    Serial.println("[UPLOAD] Queue full, dropping frame");
    free(copy);
    return false;
  }
  return true;
}



// ── WiFi connect ──────────────────────────────────────────────────────────────
static bool connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi connect");

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_TIMEOUT_MS) {
    delay(100); Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    return true;
  }

  Serial.println("WiFi failed");
  return false;
}

// ── checkSleepMode — fetches toggle state from server ────────────────────────
static void checkSleepMode() {
  if (WiFi.status() != WL_CONNECTED) return; // keep last known value

  WiFiClient client;
  client.setTimeout(2000);
  if (!client.connect(LAPTOP_IP, LAPTOP_PORT)) return;

  client.print("GET /api/sleep-mode HTTP/1.1\r\n");
  client.print("Host: " + String(LAPTOP_IP) + "\r\n");
  client.print("Connection: close\r\n\r\n");

  unsigned long timeout = millis() + 2000;
  while (!client.available() && millis() < timeout) delay(5);

  while (client.available()) {
    String line = client.readStringUntil('\n');
    if (line.indexOf("enabled") >= 0) {
      sleepEnabled = line.indexOf("true") >= 0;
      break;
    }
  }
  client.stop();
  Serial.printf("Sleep mode: %s\n", sleepEnabled ? "ON" : "OFF");
}

// ── goToDeepSleep — helper ────────────────────────────────────────────────────
static void goToDeepSleep() {
  Serial.printf("Sleeping %ds (boot #%d, active for %lums)\n",
    SLEEP_SECONDS, bootCount, millis());
  Serial.flush();
  esp_sleep_enable_timer_wakeup((uint64_t)SLEEP_SECONDS * 1000000ULL);
  esp_deep_sleep_start();
}

// ── setup — runs on every wake from deep sleep ────────────────────────────────
void setup() {
  bootCount++;
  Serial.begin(BAUD);
  Serial.printf("\n=== Boot #%d ===\n", bootCount);

  Wire.begin();
  SPI.begin();
  myCamera.begin();

  // Restore system clock from RTC memory so time() works immediately
  if (rtcTime > 0) {
    struct timeval tv = { .tv_sec = rtcTime + SLEEP_SECONDS, .tv_usec = 0 };
    settimeofday(&tv, nullptr);
    Serial.println("Clock restored from RTC memory");
  }

  bool wifiOk = connectWiFi();

  if (wifiOk) {
    // Re-sync NTP on first boot or every NTP_RESYNC_BOOTS boots
    bool needNtp = (rtcTime == 0) || (bootCount % NTP_RESYNC_BOOTS == 1);
    if (needNtp) {
      configTime(0, 0, "pool.ntp.org");
      Serial.print("NTP sync");
      struct tm timeinfo;
      if (getLocalTime(&timeinfo, 5000)) {
        time(&rtcTime);
        Serial.println(" ✅");
      } else {
        Serial.println(" ❌ (using RTC time)");
      }
    } else {
      // Save current time for next boot
      time(&rtcTime);
      Serial.printf("NTP skipped (boot %d, next sync at boot %d)\n",
        bootCount, ((bootCount / NTP_RESYNC_BOOTS) + 1) * NTP_RESYNC_BOOTS + 1);
    }
  }

  // Create upload queue and task
  uploadQueue = xQueueCreate(2, sizeof(UploadJob));
  xTaskCreatePinnedToCore(uploadTask, "uploadTask", 8192, nullptr, 1, &uploadTaskHandle, 0);

  // Run inference (single shot)
  inferenceSetup();
  runInference();

  // Wait for upload queue to drain
  Serial.print("Waiting for upload...");
  unsigned long waitStart = millis();
  while (uxQueueMessagesWaiting(uploadQueue) > 0 && millis() - waitStart < UPLOAD_WAIT_MS) {
    delay(25);
  }
  Serial.println(" done");

  // Check sleep toggle from frontend
  checkSleepMode();

  if (sleepEnabled) {
    goToDeepSleep();  // never returns
  }

  // Sleep is OFF — stay awake and keep inferring continuously
  Serial.println("Sleep disabled — running continuously");
}

void loop() {
  // Only reached when sleep mode is OFF
  server.handleClient();

  static unsigned long lastInference = 0;
  if (millis() - lastInference >= (SLEEP_SECONDS * 1000)) {
    lastInference = millis();
    runInference();

    // Re-check toggle each cycle so turning sleep back on works immediately
    checkSleepMode();
    if (sleepEnabled) {
      // Drain upload queue before sleeping
      Serial.print("Sleep re-enabled, waiting for upload...");
      unsigned long w = millis();
      while (uxQueueMessagesWaiting(uploadQueue) > 0 && millis() - w < UPLOAD_WAIT_MS) {
        delay(25);
      }
      Serial.println(" done");
      goToDeepSleep();  // never returns
    }
  }
}