"""
Geospatial Surface Recognition System
======================================
CNN-based terrain image classifier built with TensorFlow/Keras.
Dataset: https://www.kaggle.com/datasets/atharv1610/terrain-recognition
Achieved ~89% validation accuracy over 20 epochs.

Authors:
    Supreet Kumar Patel   
    Prakhar Patel    
    Amrutha Jampala     
    Abhiamrit Veera           
    S Priyanshu Nayak            
    Rashu Shankar    

Guide: Prof. Dr. Ajit Kumar Pasayat
Institution: KIIT Deemed to be University, Bhubaneswar
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, LeakyReLU
)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = "dataset"          # Folder with class subfolders
IMG_HEIGHT  = 150
IMG_WIDTH   = 150
BATCH_SIZE  = 32
EPOCHS      = 20
VAL_SPLIT   = 0.2
SEED        = 123
MODEL_PATH  = "terrain_classifier.h5"

# Class names — must match your dataset folder names (alphabetical order)
CLASS_NAMES = ["Grassy", "Marshy", "Rocky", "Sandy"]


# ─────────────────────────────────────────────
# STEP 1: LOAD & SPLIT DATASET
# ─────────────────────────────────────────────
def load_datasets(data_dir):
    """Load training and validation datasets from directory."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print("Classes found:", train_ds.class_names)
    return train_ds, val_ds


# ─────────────────────────────────────────────
# STEP 2: NORMALIZE & OPTIMIZE PIPELINE
# ─────────────────────────────────────────────
def preprocess(train_ds, val_ds):
    """Normalize pixel values to [0, 1] and cache/prefetch for performance."""

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


# ─────────────────────────────────────────────
# STEP 3: BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_classes=4):
    """
    CNN architecture matching terrain_classifier.h5.

    Blocks:   3x (Conv2D -> LeakyReLU -> MaxPooling2D)
    Head:     Flatten -> Dense(512) -> LeakyReLU -> Dropout(0.5) -> Dense(num_classes)
    Loss:     SparseCategoricalCrossentropy(from_logits=True)
    Optimizer: Adam
    """

    model = Sequential([
        # Block 1: 64 filters
        Conv2D(64, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        LeakyReLU(negative_slope=0.01),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2: 128 filters
        Conv2D(128, (3, 3)),
        LeakyReLU(negative_slope=0.01),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3: 256 filters
        Conv2D(256, (3, 3)),
        LeakyReLU(negative_slope=0.01),
        MaxPooling2D(pool_size=(2, 2)),

        # Classifier head
        Flatten(),
        Dense(512),
        LeakyReLU(negative_slope=0.01),
        Dropout(0.5),
        Dense(num_classes)   # No softmax — using from_logits=True in loss
    ], name="terrain_classifier")

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


# ─────────────────────────────────────────────
# STEP 4: TRAIN
# ─────────────────────────────────────────────
def train_model(model, train_ds, val_ds):
    """Train model for EPOCHS and return history."""
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    return history


# ─────────────────────────────────────────────
# STEP 5: EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, val_ds):
    """Print loss, accuracy, classification report, and confusion matrix."""

    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"\nValidation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    val_predictions = model.predict(val_ds)
    val_labels      = np.concatenate([y for x, y in val_ds], axis=0)
    val_predictions = np.argmax(val_predictions, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(val_labels, val_predictions, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, val_predictions))


# ─────────────────────────────────────────────
# STEP 6: PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(history):
    """Plot and save accuracy & loss graphs."""

    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc,     label='Train')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss,     label='Train')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
    print("Plot saved as training_history.png")


# ─────────────────────────────────────────────
# STEP 7: SAVE & LOAD MODEL
# ─────────────────────────────────────────────
def save_model(model, path=MODEL_PATH):
    model.save(path)
    print(f"Model saved: {path}")


def load_pretrained(path=MODEL_PATH):
    model = tf.keras.models.load_model(path)
    print("Model loaded successfully!")
    model.summary()
    return model


# ─────────────────────────────────────────────
# STEP 8: PREDICT A SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_image(model, image_path, class_names=CLASS_NAMES):
    """
    Predict terrain class of a single image.

    Args:
        model      : Loaded Keras model
        image_path : Path to image (JPEG or PNG)
        class_names: List of class label names

    Returns:
        predicted class name and confidence score
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array  = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    predictions     = model.predict(img_array)
    probabilities   = tf.nn.softmax(predictions[0]).numpy()
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence      = probabilities[predicted_index]

    print(f"\nAll probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class} ({confidence:.2%} confidence)")

    return predicted_class, confidence


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("  Geospatial Surface Recognition System")
    print("=" * 50)

    # ── Option A: Train from scratch ──
    print("\n[1] Loading dataset...")
    train_ds, val_ds = load_datasets(DATA_DIR)

    print("\n[2] Preprocessing...")
    train_ds, val_ds = preprocess(train_ds, val_ds)

    print("\n[3] Building model...")
    model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    print("\n[4] Training...")
    history = train_model(model, train_ds, val_ds)

    print("\n[5] Evaluating...")
    evaluate_model(model, val_ds)

    print("\n[6] Plotting history...")
    plot_history(history)

    print("\n[7] Saving model...")
    save_model(model)

    # ── Option B: Load existing model and predict ──
    # model = load_pretrained("terrain_classifier.h5")
    # predict_image(model, "path/to/your/image.jpg")

    print("\n[8] Predict a new image:")
    image_path = input("Enter image path: ")
    predict_image(model, image_path)
    preprocess 



