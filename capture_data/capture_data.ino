#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <SPI.h>
#include "ardCam.h"

const char* SSID     = "Proximus-Home-550214";
const char* PASSWORD = "ns5xn9u7bfym4nen";

WebServer server(80);

void handleCapture() {
  CamStatus st = myCamera.takePicture(CAM_IMAGE_MODE_96X96, CAM_IMAGE_PIX_FMT_RGB565);
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
  Serial.printf("Sent %lu bytes\n", sent);
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
  Serial.println("HTTP server started, GET /capture to take a picture");
}

void loop() {
  server.handleClient();
}