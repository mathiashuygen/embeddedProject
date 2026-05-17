"""
Balanced classifier for ESP32-S3 - multiple model sizes.
Allows trade-off between speed and accuracy.
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import subprocess

# ── Config ────────────────────────────────────────────────────
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 50
NOT_ALLOWED_DIR = "data/not_allowed/"
ALLOWED_DIR = "data/allowed/"
# ─────────────────────────────────────────────────────────────


def load_images_from_folder(folder, label, label_name, max_images=None):
    """Load 96x96 images directly - no cropping needed."""
    images = []
    labels = []

    files = sorted(os.listdir(folder))
    if max_images:
        files = files[:max_images]

    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print(f"Loaded {len(images)} {label_name} images from {folder}")
    return np.array(images), np.array(labels)


def augment_data_intelligent(images, labels):
    """Smarter augmentation for small datasets."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.7, 1.3],
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="reflect",
    )

    augmented_images = []
    augmented_labels = []
    num_augmentations = 5 if len(images) < 50 else 3

    for i in range(len(images)):
        img = images[i].reshape((1, IMG_SIZE, IMG_SIZE, 3))
        label = labels[i]

        augmented_images.append(images[i])
        augmented_labels.append(label)

        count = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            count += 1
            if count >= num_augmentations:
                break

    return np.array(augmented_images), np.array(augmented_labels)


def build_efficient_model(img_size):
    """Fast model: 0.3–0.6s inference, lower accuracy."""
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inp)

    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    print(f"Efficient model parameters: {model.count_params():,}")
    return model


def build_hybrid_model(img_size):
    """Balanced model: 0.6–1.0s inference, good accuracy."""
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inp)

    x = tf.keras.layers.Conv2D(24, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(48, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(96, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    print(f"Hybrid model parameters: {model.count_params():,}")
    return model


def build_medium_model(img_size):
    """Slower model: 1.0–1.8s inference, best accuracy."""
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inp)

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    print(f"Medium model parameters: {model.count_params():,}")
    return model


def build_ensemble_model(img_size):
    """Ensemble of 3 small models – highest accuracy but slowest."""
    from tensorflow.keras import layers

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    normalized = layers.Rescaling(1.0 / 255)(inputs)

    branch1 = layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(
        normalized
    )
    branch1 = layers.MaxPooling2D(2)(branch1)
    branch1 = layers.Conv2D(32, 3, activation="relu")(branch1)
    branch1 = layers.GlobalAveragePooling2D()(branch1)

    branch2 = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(
        normalized
    )
    branch2 = layers.DepthwiseConv2D(3, activation="relu")(branch2)
    branch2 = layers.Conv2D(32, 1, activation="relu")(branch2)
    branch2 = layers.GlobalAveragePooling2D()(branch2)

    branch3 = layers.Conv2D(8, 3, strides=2, activation="relu")(normalized)
    branch3 = layers.Conv2D(16, 3, activation="relu")(branch3)
    branch3 = layers.GlobalAveragePooling2D()(branch3)

    merged = layers.Concatenate()([branch1, branch2, branch3])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(32, activation="relu")(merged)
    output = layers.Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    print(f"Ensemble model parameters: {model.count_params():,}")
    return model


def export_optimized_tflite(model, calibration_images):
    """Export to quantized int8 TFLite for ESP32."""

    def representative_dataset():
        for i in range(0, min(100, len(calibration_images)), BATCH_SIZE):
            batch = calibration_images[i : i + BATCH_SIZE]
            yield [batch.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    size_kb = len(tflite_model) / 1024
    print(f"\n📦 Quantized model size: {len(tflite_model)} bytes ({size_kb:.1f} KB)")
    if size_kb < 200:
        print("   Expected inference: 0.3-0.6 seconds")
    elif size_kb < 500:
        print("   Expected inference: 0.6-1.2 seconds")
    else:
        print("   Expected inference: 1-2 seconds")
    return tflite_model


def main():
    print("Select model for ESP32-S3:")
    print("1. Efficient  (fast, 0.3–0.6s, lower accuracy)")
    print("2. Hybrid     (balanced, 0.6–1.0s, good accuracy)")
    print("3. Medium     (slower, 1.0–1.8s, best accuracy)")
    print("4. Ensemble   (slowest, >2s, highest accuracy)")
    choice = input("\nChoice (1-4) [default: 3]: ").strip() or "3"

    print("\n=== Loading datasets ===")
    allowed_imgs, allowed_labels = load_images_from_folder(ALLOWED_DIR, 0, "allowed")
    not_allowed_imgs, not_allowed_labels = load_images_from_folder(
        NOT_ALLOWED_DIR, 1, "not allowed"
    )

    if len(allowed_imgs) < 2 or len(not_allowed_imgs) < 2:
        print("ERROR: Need at least 2 images per class!")
        sys.exit(1)

    X = np.concatenate([allowed_imgs, not_allowed_imgs])
    y = np.concatenate([allowed_labels, not_allowed_labels])

    print(f"\nDataset summary:")
    print(f"  Allowed: {len(allowed_imgs)} images")
    print(f"  Not allowed: {len(not_allowed_imgs)} images")
    print(f"  Total: {len(X)} images")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if len(X_train) < 100:
        print(
            f"\n⚠️ Small dataset ({len(X_train)} training images). Applying augmentation..."
        )
        X_train, y_train = augment_data_intelligent(X_train, y_train)
        print(f"   Augmented training set: {len(X_train)} images")
    else:
        print(
            f"\n✅ Dataset size OK ({len(X_train)} training images). Skipping augmentation."
        )

    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")

    if choice == "1":
        model = build_efficient_model(IMG_SIZE)
    elif choice == "2":
        model = build_hybrid_model(IMG_SIZE)
    elif choice == "3":
        model = build_medium_model(IMG_SIZE)
    else:
        model = build_ensemble_model(IMG_SIZE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, verbose=1, monitor="val_loss"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]

    print("\n=== Training ===")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    print("\n=== Evaluation ===")
    results = model.evaluate(X_val, y_val, verbose=0)
    loss, acc = results[0], results[1]
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc:.4f}")
    if len(results) > 2:
        for i in range(2, len(results)):
            metric_name = (
                model.metrics_names[i]
                if i < len(model.metrics_names)
                else f"metric_{i}"
            )
            print(f"{metric_name}: {results[i]:.4f}")

    # Save labels
    with open("labels.txt", "w") as f:
        f.write("allowed\nnot_allowed")

    print("\n=== Exporting TFLite model ===")
    tflite_model = export_optimized_tflite(model, X_train[:100])
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    subprocess.run(["xxd", "-i", "model.tflite", "model_data.h"])

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig("training_history.png")
    print("✅ Saved training_history.png")

    print("\n✅ Done! Files written:")
    print("  model.tflite")
    print("  model_data.h")
    print("  labels.txt")
    print("  training_history.png")


if __name__ == "__main__":
    main()
