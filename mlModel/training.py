"""
Autoencoder-based anomaly detector for dog toy detection.
Trains on "allowed" images only. High reconstruction error = not allowed.

Usage:
    python train.py

Output:
    model.tflite        — quantized TFLite model
    model_data.h        — C header with model as byte array
    threshold.txt       — reconstruction error threshold
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────
IMG_SIZE = 96  # must match camera capture size
BATCH_SIZE = 16
EPOCHS = 100
DATA_DIR = "../allowed"  # adjust if needed
# ─────────────────────────────────────────────────────────────


def load_images(path):
    images = []
    for fname in sorted(os.listdir(path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(path, fname)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
    print(f"Loaded {len(images)} images from {path}")
    return np.array(images)


def build_autoencoder(img_size):
    inp = tf.keras.Input(shape=(img_size, img_size, 3))

    # Encoder
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", strides=2)(inp)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", strides=2)(x)
    encoded = tf.keras.layers.Conv2D(
        4, 3, activation="relu", padding="same", strides=2
    )(x)

    # Decoder — use Conv2DTranspose instead of UpSampling2D
    x = tf.keras.layers.Conv2DTranspose(
        4, 3, activation="relu", padding="same", strides=2
    )(encoded)
    x = tf.keras.layers.Conv2DTranspose(
        8, 3, activation="relu", padding="same", strides=2
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        16, 3, activation="relu", padding="same", strides=2
    )(x)
    decoded = tf.keras.layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inp, decoded, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def compute_threshold(model, images, percentile=95):
    """Set threshold at the Nth percentile of training reconstruction errors."""
    preds = model.predict(images, verbose=0)
    errors = np.mean((images - preds) ** 2, axis=(1, 2, 3))
    threshold = float(np.percentile(errors, percentile))
    print(
        f"Reconstruction errors — min: {errors.min():.4f}, "
        f"max: {errors.max():.4f}, mean: {errors.mean():.4f}"
    )
    print(f"Threshold (p{percentile}): {threshold:.4f}")
    return threshold


def export_tflite(model, images):
    """Quantize and export to TFLite using float16."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"Saved model.tflite ({len(tflite_model)} bytes)")
    return tflite_model


def export_header(tflite_model):
    """Convert tflite binary to C header."""
    hex_array = ", ".join(f"0x{b:02x}" for b in tflite_model)
    header = f"""// Auto-generated — do not edit
#pragma once
#include <stdint.h>

const unsigned int model_tflite_len = {len(tflite_model)};
const uint8_t model_tflite[] = {{
  {hex_array}
}};
"""
    with open("model_data.h", "w") as f:
        f.write(header)
    print("Saved model_data.h")


def main():
    print(f"TensorFlow {tf.__version__}")

    images = load_images(DATA_DIR)
    if len(images) < 10:
        print("Not enough images — need at least 10")
        sys.exit(1)

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    model = build_autoencoder(IMG_SIZE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    model.fit(
        train_imgs,
        train_imgs,
        validation_data=(val_imgs, val_imgs),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    threshold = compute_threshold(model, train_imgs)
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))

    tflite_model = export_tflite(model, train_imgs)
    export_header(tflite_model)

    print("\nDone! Files written:")
    print("  model.tflite")
    print("  model_data.h")
    print("  threshold.txt")
    print(f"\nUse threshold {threshold:.4f} in your ESP32 code.")


if __name__ == "__main__":
    main()
