"""
Binary classifier for dog toy detection.
Trains on "allowed" vs "not allowed" images.

Usage:
    python train_classifier.py

Output:
    model.tflite        — quantized TFLite model for ESP32
    model_data.h        — C header with model as byte array
    labels.txt          — class labels
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ── Config ────────────────────────────────────────────────────
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 50
ALLOWED_DIR = "../allowed/"
NOT_ALLOWED_DIR = "../not_allowed/"
# ─────────────────────────────────────────────────────────────


def center_crop_and_resize(img, size):
    """Center crop to square then resize to size x size."""
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def load_images_from_folder(folder, label, label_name, max_images=None):
    """Load images from a folder and assign label."""
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
            img = center_crop_and_resize(img, IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print(f"Loaded {len(images)} {label_name} images from {folder}")
    return np.array(images), np.array(labels)


def augment_data(images, labels):
    """Data augmentation for better generalization."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Create augmented dataset
    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        img = images[i].reshape((1, IMG_SIZE, IMG_SIZE, 3))
        label = labels[i]

        # Add original
        augmented_images.append(images[i])
        augmented_labels.append(label)

        # Add 3 augmented versions
        count = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            count += 1
            if count >= 3:
                break

    return np.array(augmented_images), np.array(augmented_labels)


def build_classifier(img_size):
    """Build CNN classifier for allowed/not allowed."""
    inp = tf.keras.Input(shape=(img_size, img_size, 3))

    # Conv block 1
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Conv block 2
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Conv block 3
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Global pooling instead of flatten for smaller model
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output: 0 = allowed, 1 = not allowed
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, output, name="toy_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def export_tflite_quantized(model, calibration_images):
    """Export to quantized int8 TFLite for ESP32."""

    # Quantization calibration
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

    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"Saved quantized model.tflite ({len(tflite_model)} bytes)")
    return tflite_model


def export_header(tflite_model):
    """Convert tflite binary to C header using xxd."""
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    # Use xxd command to generate header
    import subprocess

    subprocess.run(["xxd", "-i", "model.tflite", "model_data.h"])
    print("Saved model_data.h via xxd")


def main():
    print(f"TensorFlow {tf.__version__}")

    # Load datasets
    print("\n=== Loading datasets ===")
    allowed_imgs, allowed_labels = load_images_from_folder(ALLOWED_DIR, 0, "allowed")
    not_allowed_imgs, not_allowed_labels = load_images_from_folder(
        NOT_ALLOWED_DIR, 1, "not allowed"
    )

    if len(allowed_imgs) < 10 or len(not_allowed_imgs) < 10:
        print("ERROR: Need at least 10 images per class!")
        sys.exit(1)

    # Combine datasets
    X = np.concatenate([allowed_imgs, not_allowed_imgs])
    y = np.concatenate([allowed_labels, not_allowed_labels])

    print(f"\nDataset summary:")
    print(f"  Allowed: {len(allowed_imgs)} images")
    print(f"  Not allowed: {len(not_allowed_imgs)} images")
    print(f"  Total: {len(X)} images")

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply data augmentation to training set only
    print("\n=== Applying data augmentation ===")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    print(f"Training set after augmentation: {len(X_train_aug)} images")

    # Handle class imbalance
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train_aug), y=y_train_aug
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")

    # Build and train model
    print("\n=== Building model ===")
    model = build_classifier(IMG_SIZE)
    model.summary()

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
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Evaluate
    print("\n=== Evaluation ===")
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save labels
    with open("labels.txt", "w") as f:
        f.write("allowed\nnot_allowed")

    # Export model (use training images for calibration)
    print("\n=== Exporting TFLite model ===")
    tflite_model = export_tflite_quantized(model, X_train[:100])
    export_header(tflite_model)

    print("\n✅ Done! Files written:")
    print("  model.tflite")
    print("  model_data.h")
    print("  labels.txt")

    # Plot training history
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["loss"], label="train")
        axes[0].plot(history.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(history.history["accuracy"], label="train")
        axes[1].plot(history.history["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()

        plt.savefig("training_history.png")
        print("Saved training_history.png")
    except:
        pass


if __name__ == "__main__":
    main()
