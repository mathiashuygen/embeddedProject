#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <SPI.h>
#include <driver/i2s.h>
#include "ardCam.h"
#include <WiFiUdp.h>

const char* SSID     = "Proximus-Home-550214";
const char* PASSWORD = "ns5xn9u7bfym4nen";
const char* LAPTOP_IP = "192.168.129.125";  
const int LAPTOP_PORT = 8080;


WebServer server(80);

WiFiClient httpClient;

#define INFERENCE_INTERVAL_MS 2000



bool testLaptopConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return false;
  }
  
  Serial.printf("Testing connection to %s:8081...\n", LAPTOP_IP);
  
  WiFiClient testClient;
  if (testClient.connect(LAPTOP_IP, 8081, 1000)) { // 1 second timeout
    Serial.println("✅ Laptop connection successful!");
    testClient.stop();
    return true;
  } else {
    Serial.println("❌ Cannot connect to laptop");
    Serial.println("   Check:");
    Serial.println("   1. Is the laptop running the Docker containers?");
    Serial.println("   2. Is the laptop's firewall blocking ports 8080/8081?");
    Serial.println("   3. Is the IP address correct?");
    Serial.printf("   4. Can you ping %s from this device?\n", LAPTOP_IP);
    return false;
  }
}

// Send image via HTTP POST
void sendImageToLaptop() {
  if (WiFi.status() != WL_CONNECTED) return;
  
  // Capture image
  CamStatus st = myCamera.takePicture(CAM_IMAGE_MODE_QVGA, CAM_IMAGE_PIX_FMT_JPG);
  if (st != CAM_ERR_SUCCESS) {
    Serial.printf("Capture failed for sending: %d\n", (int)st);
    return;
  }
  
  uint32_t total = myCamera.getTotalLength();
  if (total == 0) return;
  
  // Connect to laptop HTTP server
  WiFiClient client;
  if (!client.connect(LAPTOP_IP, 8081)) {
    Serial.println("Connection to laptop failed");
    return;
  }
  
  // Send multipart form data
  String boundary = "----ESP32Boundary";
  String header = "POST /upload-image HTTP/1.1\r\n";
  header += "Host: " + String(LAPTOP_IP) + ":8081\r\n";
  header += "Content-Type: multipart/form-data; boundary=" + boundary + "\r\n";
  header += "Content-Length: ";
  
  // Calculate total size
  uint32_t imageSize = total;
  uint32_t totalSize = imageSize + 500; // Approximate header size
  
  header += String(totalSize) + "\r\n\r\n";
  client.print(header);
  
  // Send boundary and image data
  client.print("--" + boundary + "\r\n");
  client.print("Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n");
  client.print("Content-Type: image/jpeg\r\n\r\n");
  
  // Send image in chunks
  uint8_t buf[CHUNK];
  uint32_t sent = 0;
  while (sent < total) {
    uint8_t toRead = (total - sent) > CHUNK ? CHUNK : (total - sent);
    uint8_t n = myCamera.readBuff(buf, toRead);
    if (n == 0) break;
    client.write(buf, n);
    sent += n;
  }
  
  client.print("\r\n--" + boundary + "--\r\n");
  client.stop();
  Serial.println("Image sent to laptop");
}


void handleCapture() {
  CamStatus st = myCamera.takePicture(CAM_IMAGE_MODE_QVGA, CAM_IMAGE_PIX_FMT_JPG);
  if (st != CAM_ERR_SUCCESS) {
    server.send(500, "text/plain", "CAM_ERR=" + String((int)st));
    return;
  }

  uint32_t total = myCamera.getTotalLength();
  if (total == 0) {
    server.send(500, "text/plain", "CAM_EMPTY");
    return;
  }

  server.setContentLength(total);
  server.send(200, "image/jpeg", "");

  uint8_t buf[CHUNK];
  uint32_t sent = 0;
  while (sent < total) {
    uint8_t toRead = (uint32_t)(total - sent) > CHUNK ? CHUNK : (uint8_t)(total - sent);
    uint8_t n = myCamera.readBuff(buf, toRead);
    if (n == 0) { Serial.println("READ_ERR"); break; }
    server.sendContent((const char*)buf, n);
    sent += n;
  }
  Serial.printf("Sent %lu bytes over WiFi\n", sent);
}

void setup() {
  Serial.begin(BAUD);
  Serial.println("BOOT");

  Wire.begin();
  SPI.begin();

  CamStatus rc = myCamera.begin();
  Serial.print("CAM_BEGIN=");
  Serial.println((int)rc);

  WiFi.begin(SSID, PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP: ");
  Serial.println(WiFi.localIP());

  server.on("/capture", HTTP_GET, handleCapture);
  server.begin();
  Serial.println("HTTP server started");

  inferenceSetup();
}

void loop() {
  server.handleClient();

  static unsigned long lastInference = 0;
  if (millis() - lastInference >= INFERENCE_INTERVAL_MS) {
    lastInference = millis();
    runInference();
  }
}