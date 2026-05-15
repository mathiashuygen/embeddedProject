
#ifdef Bool
#undef Bool
#endif
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include "ardCam.h"
#include "audio.h"
#include "model_data.h"

// ── Config ────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE (800 * 1024)

// ── I2S / Audio ───────────────────────────────────────────────
#define PIN_I2S_BCLK 11
#define PIN_I2S_LRCK 10
#define PIN_I2S_DOUT 9

static const i2s_port_t I2S_PORT = I2S_NUM_0;
static const int SAMPLE_RATE = 16000;
static bool i2sReady = false;

// ── PSRAM buffers ─────────────────────────────────────────────
static uint8_t* tensorArena = nullptr;
static uint8_t* rgbBuf = nullptr;
static uint8_t* interpMem = nullptr;

// AllOpsResolver as static — avoids placement new restriction
// It lives in internal RAM but is small enough (~2KB)
static tflite::AllOpsResolver allOpsResolver;

static tflite::MicroInterpreter* interpreter = nullptr;
static const tflite::Model* tflModel = nullptr;

static bool inferenceReady = false;


// store the current inference image
static uint8_t* inferenceImageBuffer = nullptr;
static uint32_t inferenceImageSize = 0;

// ── Audio ─────────────────────────────────────────────────────

static void i2sInit() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num = PIN_I2S_BCLK,
    .ws_io_num = PIN_I2S_LRCK,
    .data_out_num = PIN_I2S_DOUT,
    .data_in_num = I2S_PIN_NO_CHANGE
  };
  if (i2s_driver_install(I2S_PORT, &cfg, 0, nullptr) != ESP_OK) {
    Serial.println("I2S install failed");
    return;
  }
  if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) {
    Serial.println("I2S pin set failed");
    return;
  }
  i2s_zero_dma_buffer(I2S_PORT);
  i2sReady = true;
  Serial.println("I2S ready");
}

void playLeaveIt() {
  if (!i2sReady) return;
  const int16_t* mono = (const int16_t*)voice_raw;
  const size_t mono_samples = voice_raw_len / 2;
  static int16_t stereo[256 * 2];
  size_t i = 0;
  while (i < mono_samples) {
    size_t frames = 256;
    if (i + frames > mono_samples) frames = mono_samples - i;
    for (size_t f = 0; f < frames; f++) {
      stereo[2 * f] = mono[i + f];
      stereo[2 * f + 1] = mono[i + f];
    }
    size_t written = 0;
    i2s_write(I2S_PORT, stereo, frames * 2 * sizeof(int16_t), &written, portMAX_DELAY);
    i += frames;
  }
}

// ── Setup ─────────────────────────────────────────────────────

void inferenceSetup() {
  Serial.printf("Free heap: %lu, Free PSRAM: %lu\n",
                ESP.getFreeHeap(), ESP.getFreePsram());

  Serial.println("Allocating tensorArena...");
  tensorArena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
  if (!tensorArena) {
    Serial.println("FAILED");
    return;
  }
  Serial.println("OK");

  Serial.println("Allocating rgbBuf...");
  rgbBuf = (uint8_t*)heap_caps_aligned_alloc(16, 96 * 96 * 2, MALLOC_CAP_SPIRAM);
  if (!rgbBuf) {
    Serial.println("FAILED");
    return;
  }
  Serial.println("OK");

  Serial.println("Loading model...");
  tflModel = tflite::GetModel(model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Schema mismatch: %d vs %d\n",
                  tflModel->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  Serial.println("OK");

  Serial.println("Allocating interpreter...");
  interpMem = (uint8_t*)heap_caps_aligned_alloc(16,
                                                sizeof(tflite::MicroInterpreter), MALLOC_CAP_SPIRAM);
  if (!interpMem) {
    Serial.println("FAILED");
    return;
  }
  interpreter = new (interpMem) tflite::MicroInterpreter(
    tflModel, allOpsResolver, tensorArena, TENSOR_ARENA_SIZE);
  Serial.println("OK");

  Serial.println("Allocating tensors...");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FAILED");
    return;
  }
  Serial.printf("OK — arena used: %u / %u bytes\n",
                interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);

  TfLiteTensor* input = interpreter->input(0);
  if (!input) {
    Serial.println("Input tensor null!");
    return;
  }
  Serial.printf("Input: ptr=%p bytes=%d type=%d\n",
                input->data.raw, input->bytes, input->type);

  TfLiteTensor* output = interpreter->output(0);
  if (!output) {
    Serial.println("Output tensor null!");
    return;
  }
  Serial.printf("Output: ptr=%p bytes=%d type=%d\n",
                output->data.raw, output->bytes, output->type);

  Serial.printf("Free heap: %lu, Free PSRAM: %lu\n",
                ESP.getFreeHeap(), ESP.getFreePsram());

  i2sInit();
  inferenceReady = true;
  Serial.println("=== Inference setup complete ===");
}

// ── Inference ─────────────────────────────────────────────────
void runInference() {
  if (!inferenceReady) {
    Serial.println("Not ready, skipping");
    return;
  }

  // Capture ONE image for inference (96x96 RGB565)
  CamStatus st = myCamera.takePicture(CAM_IMAGE_MODE_96X96, CAM_IMAGE_PIX_FMT_RGB565);
  if (st != CAM_ERR_SUCCESS) {
    Serial.printf("Capture failed: %d\n", (int)st);
    return;
  }

  uint32_t total = myCamera.getTotalLength();
  if (total == 0 || total > 96 * 96 * 2) {
    Serial.printf("Bad frame size: %lu\n", total);
    return;
  }

  // Read RGB565 into PSRAM rgbBuf (for inference)
  uint32_t bytesRead = 0;
  while (bytesRead < total) {
    uint8_t toRead = (total - bytesRead > CHUNK) ? CHUNK : (uint8_t)(total - bytesRead);
    uint8_t n = myCamera.readBuff(rgbBuf + bytesRead, toRead);
    if (n == 0) {
      Serial.println("READ_ERR");
      return;
    }
    bytesRead += n;
  }

  // STORE the image for later sending (convert to JPEG for smaller size)
  // First, free previous buffer if exists
  if (inferenceImageBuffer) {
    free(inferenceImageBuffer);
    inferenceImageBuffer = nullptr;
  }

  // Capture a JPEG version of the SAME scene for sending to laptop
  // Note: The camera is still pointing at the same scene
  st = myCamera.takePicture(CAM_IMAGE_MODE_QVGA, CAM_IMAGE_PIX_FMT_JPG);
  if (st == CAM_ERR_SUCCESS) {
    inferenceImageSize = myCamera.getTotalLength();
    if (inferenceImageSize > 0 && inferenceImageSize < 100 * 1024) {  // Max 100KB
      inferenceImageBuffer = (uint8_t*)malloc(inferenceImageSize);
      if (inferenceImageBuffer) {
        uint32_t jpgRead = 0;
        while (jpgRead < inferenceImageSize) {
          uint8_t toRead = (inferenceImageSize - jpgRead > CHUNK) ? CHUNK : (uint8_t)(inferenceImageSize - jpgRead);
          uint8_t n = myCamera.readBuff(inferenceImageBuffer + jpgRead, toRead);
          if (n == 0) break;
          jpgRead += n;
        }
        Serial.printf("Stored %lu bytes JPEG for sending\n", inferenceImageSize);
      }
    }
  }

  // Perform inference on the RGB565 image (already in rgbBuf)
  TfLiteTensor* input = interpreter->input(0);
  uint8_t* dst = input->data.uint8;

  for (uint32_t px = 0; px < 96 * 96; px++) {
    uint16_t rgb565 = ((uint16_t)rgbBuf[px * 2] << 8) | rgbBuf[px * 2 + 1];
    uint8_t r = (rgb565 >> 11) & 0x1F;
    r = (r << 3) | (r >> 2);
    uint8_t g = (rgb565 >> 5) & 0x3F;
    g = (g << 2) | (g >> 4);
    uint8_t b = rgb565 & 0x1F;
    b = (b << 3) | (b >> 2);

    dst[px * 3 + 0] = r;
    dst[px * 3 + 1] = g;
    dst[px * 3 + 2] = b;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  float probability = output->data.uint8[0] / 255.0f;
  bool isNotAllowed = probability > 0.5;

  Serial.printf("Probability: %.2f%% — %s\n",
                probability * 100,
                isNotAllowed ? "NOT ALLOWED" : "Allowed");

  // Send result AND the captured image over UDP + HTTP
  if (WiFi.status() == WL_CONNECTED && inferenceImageBuffer && inferenceImageSize > 0) {
    sendImageToLaptopWithResult(probability, isNotAllowed);
  }
}





bool sendImageToLaptopWithResult(float probability, bool isNotAllowed) {
  if (!inferenceImageBuffer || inferenceImageSize == 0) {
    Serial.println("No image buffer to send");
    return false;
  }
  
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return false;
  }
  
  Serial.printf("Attempting HTTP POST to %s:8081...\n", LAPTOP_IP);
  
  WiFiClient client;
  
  // Set timeout
  client.setTimeout(5000);
  
  if (!client.connect(LAPTOP_IP, LAPTOP_PORT)) {
    Serial.printf("❌ HTTP connection failed to %s:8081\n", LAPTOP_IP);
    Serial.println("   Make sure the backend container is running:");
    Serial.println("   Run: docker-compose ps");
    Serial.println("   Run: docker-compose logs backend");
    return false;
  }
  
  Serial.println("✅ Connected to laptop HTTP server");
  
  String boundary = "----ESP32Boundary";
  
  // Create JSON metadata
  String jsonData = "{";
  jsonData += "\"probability\":" + String(probability, 6) + ",";
  jsonData += "\"result\":\"" + String(isNotAllowed ? "NOT_ALLOWED" : "allowed") + "\",";
  jsonData += "\"timestamp\":" + String(millis());
  jsonData += "}";
  
  // Build multipart request
  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"metadata\"\r\n";
  bodyStart += "Content-Type: application/json\r\n\r\n";
  bodyStart += jsonData + "\r\n";
  
  bodyStart += "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"inference.jpg\"\r\n";
  bodyStart += "Content-Type: image/jpeg\r\n\r\n";
  
  String bodyEnd = "\r\n--" + boundary + "--\r\n";
  
  uint32_t totalSize = bodyStart.length() + inferenceImageSize + bodyEnd.length();
  
  // Send HTTP headers
  client.print("POST /upload-inference HTTP/1.1\r\n");
  client.print("Host: " + String(LAPTOP_IP) + ":8080\r\n");
  client.print("Content-Type: multipart/form-data; boundary=" + boundary + "\r\n");
  client.print("Content-Length: " + String(totalSize) + "\r\n");
  client.print("\r\n");
  
  // Send multipart body
  client.print(bodyStart);
  
  // Send image binary data in chunks
  uint32_t sent = 0;
  while (sent < inferenceImageSize) {
    uint32_t toSend = (inferenceImageSize - sent > CHUNK) ? CHUNK : (inferenceImageSize - sent);
    int bytesWritten = client.write(inferenceImageBuffer + sent, toSend);
    if (bytesWritten <= 0) {
      Serial.println("❌ Failed to send image data");
      client.stop();
      return false;
    }
    sent += bytesWritten;
  }
  
  client.print(bodyEnd);
  
  // Wait for response
  unsigned long timeout = millis() + 3000;
  while (!client.available() && millis() < timeout) {
    delay(10);
  }
  
  String response = "";
  while (client.available()) {
    response += client.readString();
  }
  
  client.stop();
  
  if (response.length() > 0) {
    Serial.printf("✅ HTTP response received (%d bytes)\n", response.length());
    if (response.indexOf("success") > 0) {
      Serial.println("✅ Image upload successful!");
      return true;
    }
  } else {
    Serial.println("⚠️ No response from server (but image may have been sent)");
  }
  
  return false;
}