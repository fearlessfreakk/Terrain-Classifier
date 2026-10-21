"""
STEP 2: Load and Preprocess Dataset
────────────────────────────────────
Loads images from dataset/ folder, applies normalization,
optimizes the data pipeline, and displays sample images.

Usage:
    python training/step2_load_data.py
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────
DATASET_DIR = "dataset"
IMG_HEIGHT  = 150
IMG_WIDTH   = 150
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
SEED        = 123
CLASS_NAMES = ["Grassy", "Marshy", "Rocky", "Sandy"]


def load_datasets():
    print("=" * 50)
    print("  STEP 2: Loading Dataset")
    print("=" * 50)

    print("\nLoading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print("\nLoading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print(f"\n✅ Classes detected: {train_ds.class_names}")
    return train_ds, val_ds


def preprocess(train_ds, val_ds):
    print("\nNormalizing pixel values [0,255] → [0,1]...")

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    print("✅ Normalization applied.")
    return train_ds, val_ds


def optimize_pipeline(train_ds, val_ds):
    print("\nOptimizing data pipeline (cache + shuffle + prefetch)...")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .cache()
        .shuffle(buffer_size=1000)
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds = (
        val_ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    print("✅ Pipeline optimized.")
    return train_ds, val_ds


def visualize_samples(train_ds):
    print("\nSaving sample images to training/sample_images.png ...")

    plt.figure(figsize=(12, 6))
    for images, labels in train_ds.take(1):
        for i in range(min(12, len(images))):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(CLASS_NAMES[labels[i]])
            plt.axis("off")

    plt.suptitle("Sample Training Images", fontsize=14)
    plt.tight_layout()
    os.makedirs("training", exist_ok=True)
    plt.savefig("training/sample_images.png", dpi=100)
    plt.close()
    print("✅ Sample images saved to training/sample_images.png")


if __name__ == "__main__":
    train_ds, val_ds = load_datasets()
    train_ds, val_ds = preprocess(train_ds, val_ds)
    train_ds, val_ds = optimize_pipeline(train_ds, val_ds)
    visualize_samples(train_ds)

    print("\n✅ Step 2 complete. Ready for Step 3.")
    print("=" * 50)