"""
STEP 4: Train the Model
────────────────────────
Runs the full training pipeline:
  - Loads and preprocesses dataset
  - Builds and compiles the CNN
  - Trains for 20 epochs
  - Saves training history graphs
  - Saves trained model as model/terrain_classifier.h5

⚠️  This step takes 3-5 hours on CPU.
    Do NOT close the terminal while running.

Usage:
    python training/step4_train.py
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, LeakyReLU
)

# ── Configuration ───────────────────────────────────────────
DATASET_DIR = "dataset"
IMG_HEIGHT  = 150
IMG_WIDTH   = 150
BATCH_SIZE  = 32
EPOCHS      = 20
VAL_SPLIT   = 0.2
SEED        = 123
MODEL_PATH  = "model/terrain_classifier.h5"
CLASS_NAMES = ["Grassy", "Marshy", "Rocky", "Sandy"]


def load_datasets():
    print("\n[1/5] Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print(f"     Classes: {train_ds.class_names}")
    return train_ds, val_ds


def preprocess(train_ds, val_ds):
    print("\n[2/5] Preprocessing...")

    norm = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = (
        train_ds
        .map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(1000)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    print("     Normalization: [0,255] → [0,1] ✅")
    print("     Pipeline: cache + shuffle + prefetch ✅")
    return train_ds, val_ds


def build_model():
    print("\n[3/5] Building CNN model...")

    model = Sequential([

        Conv2D(64, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        LeakyReLU(negative_slope=0.01),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3)),
        LeakyReLU(negative_slope=0.01),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3)),
        LeakyReLU(negative_slope=0.01),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(512),
        LeakyReLU(negative_slope=0.01),
        Dropout(0.5),
        Dense(4)

    ], name="terrain_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print(f"     Total parameters: {model.count_params():,}")
    return model


def train_model(model, train_ds, val_ds):
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    print(f"      Batch size:    {BATCH_SIZE}")
    print(f"      Steps/epoch:   ~{5416 // BATCH_SIZE}")
    print(f"      Total updates: ~{(5416 // BATCH_SIZE) * EPOCHS}")
    print(f"\n      ⚠️  This will take 3-5 hours on CPU.\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    return history


def plot_history(history):
    print("\nSaving training graphs...")

    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_x = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_x, acc,     'b-o', label='Train Accuracy',      linewidth=2)
    plt.plot(epochs_x, val_acc, 'r-o', label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim([0.5, 1.0])

    plt.subplot(1, 2, 2)
    plt.plot(epochs_x, loss,     'b-o', label='Train Loss',      linewidth=2)
    plt.plot(epochs_x, val_loss, 'r-o', label='Validation Loss', linewidth=2)
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("training", exist_ok=True)
    plt.savefig("training/training_history.png", dpi=150)
    plt.close()

    final_train_acc = acc[-1]     * 100
    final_val_acc   = val_acc[-1] * 100
    print(f"✅ Graphs saved to training/training_history.png")
    print(f"\n   Final Training Accuracy:   {final_train_acc:.2f}%")
    print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"   Gap (overfitting check):   {final_train_acc - final_val_acc:.2f}%")


def save_model(model):
    print(f"\n[5/5] Saving model to {MODEL_PATH}...")

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model saved successfully!")
        print(f"   Path: {MODEL_PATH}")
        print(f"   Size: {size_mb:.1f} MB")
    else:
        print("❌ ERROR: Model file was not created!")


if __name__ == "__main__":
    print("=" * 55)
    print("  GEOSPATIAL SURFACE RECOGNITION SYSTEM")
    print("  Step 4: Model Training")
    print("=" * 55)

    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ ERROR: '{DATASET_DIR}/' folder not found!")
        print("   Run step1_verify_dataset.py first.")
        exit(1)

    train_ds, val_ds = load_datasets()
    train_ds, val_ds = preprocess(train_ds, val_ds)
    model            = build_model()
    history          = train_model(model, train_ds, val_ds)
    plot_history(history)
    save_model(model)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE ✅")
    print(f"  Model saved at: {MODEL_PATH}")
    print("  Run step5_evaluate.py for full evaluation report.")
    print("=" * 55)