
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
#define ANOMALY_THRESHOLD 0.0055f
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
static uint8_t* rgbBuf      = nullptr;
static uint8_t* interpMem   = nullptr;

// AllOpsResolver as static — avoids placement new restriction
// It lives in internal RAM but is small enough (~2KB)
static tflite::AllOpsResolver allOpsResolver;

static tflite::MicroInterpreter* interpreter = nullptr;
static const tflite::Model*      tflModel    = nullptr;

static bool inferenceReady = false;

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
    .bck_io_num   = PIN_I2S_BCLK,
    .ws_io_num    = PIN_I2S_LRCK,
    .data_out_num = PIN_I2S_DOUT,
    .data_in_num  = I2S_PIN_NO_CHANGE
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
      stereo[2 * f]     = mono[i + f];
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
  if (!tensorArena) { Serial.println("FAILED"); return; }
  Serial.println("OK");

  Serial.println("Allocating rgbBuf...");
  rgbBuf = (uint8_t*)heap_caps_aligned_alloc(16, 96 * 96 * 2, MALLOC_CAP_SPIRAM);
  if (!rgbBuf) { Serial.println("FAILED"); return; }
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
  if (!interpMem) { Serial.println("FAILED"); return; }
  interpreter = new(interpMem) tflite::MicroInterpreter(
    tflModel, allOpsResolver, tensorArena, TENSOR_ARENA_SIZE
  );
  Serial.println("OK");

  Serial.println("Allocating tensors...");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FAILED");
    return;
  }
  Serial.printf("OK — arena used: %u / %u bytes\n",
    interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);

  TfLiteTensor* input = interpreter->input(0);
  if (!input) { Serial.println("Input tensor null!"); return; }
  Serial.printf("Input: ptr=%p bytes=%d type=%d\n",
    input->data.raw, input->bytes, input->type);

  TfLiteTensor* output = interpreter->output(0);
  if (!output) { Serial.println("Output tensor null!"); return; }
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

  Serial.printf("Free heap: %lu\n", ESP.getFreeHeap());

  // Capture 96x96 RGB565
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

  // Read RGB565 into PSRAM rgbBuf
  uint32_t bytesRead = 0;
  while (bytesRead < total) {
    uint8_t toRead = (total - bytesRead > CHUNK) ? CHUNK : (uint8_t)(total - bytesRead);
    uint8_t n = myCamera.readBuff(rgbBuf + bytesRead, toRead);
    if (n == 0) { Serial.println("READ_ERR"); return; }
    bytesRead += n;
  }
  Serial.printf("Frame read: %lu bytes\n", bytesRead);

  // Convert RGB565 → float RGB888 directly into input tensor
  TfLiteTensor* input = interpreter->input(0);
  float* dst = input->data.f;

  for (uint32_t px = 0; px < 96 * 96; px++) {
    uint16_t rgb565 = ((uint16_t)rgbBuf[px * 2] << 8) | rgbBuf[px * 2 + 1];
    uint8_t r = (rgb565 >> 11) & 0x1F; r = (r << 3) | (r >> 2);
    uint8_t g = (rgb565 >> 5)  & 0x3F; g = (g << 2) | (g >> 4);
    uint8_t b =  rgb565        & 0x1F; b = (b << 3) | (b >> 2);
    dst[px * 3 + 0] = r / 255.0f;
    dst[px * 3 + 1] = g / 255.0f;
    dst[px * 3 + 2] = b / 255.0f;
  }
  Serial.println("Running inference...");

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }
  Serial.println("Inference done");

  // Compute MSE between input and reconstructed output
  TfLiteTensor* output = interpreter->output(0);
  float mse = 0.0f;
  uint32_t n = 96 * 96 * 3;
  for (uint32_t i = 0; i < n; i++) {
    float diff = dst[i] - output->data.f[i];
    mse += diff * diff;
  }
  mse /= n;

  Serial.printf("MSE: %.6f — %s\n",
    mse, mse > ANOMALY_THRESHOLD ? "NOT ALLOWED" : "Allowed");

  if (mse > ANOMALY_THRESHOLD) {
    playLeaveIt();
  }
}