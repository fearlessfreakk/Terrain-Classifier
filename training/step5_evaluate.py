"""
STEP 5: Evaluate the Model
───────────────────────────
Loads the saved model and runs full evaluation:
  - Overall validation accuracy and loss
  - Per-class precision, recall, F1-score
  - Confusion matrix
  - Saves evaluation report as training/evaluation_report.txt

Run this AFTER step4_train.py has completed.

Usage:
    python training/step5_evaluate.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ───────────────────────────────────────────
DATASET_DIR = "dataset"
IMG_HEIGHT  = 150
IMG_WIDTH   = 150
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
SEED        = 123
MODEL_PATH  = "model/terrain_classifier.h5"
CLASS_NAMES = ["Grassy", "Marshy", "Rocky", "Sandy"]


def load_val_dataset():
    print("Loading validation dataset...")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    norm   = tf.keras.layers.Rescaling(1.0 / 255)
    val_ds = (
        val_ds
        .map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return val_ds


def evaluate(model, val_ds):
    print("\nRunning evaluation on validation set...")

    val_loss, val_acc = model.evaluate(val_ds, verbose=1)

    print(f"\n{'=' * 45}")
    print(f"  Validation Loss:     {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Target:              85.00%")
    print(f"  Status: {'✅ PASSED' if val_acc >= 0.85 else '❌ BELOW TARGET'}")
    print(f"{'=' * 45}")

    return val_loss, val_acc


def get_predictions(model, val_ds):
    print("\nGenerating predictions...")

    all_preds  = []
    all_labels = []

    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        probs  = tf.nn.softmax(logits).numpy()
        preds  = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def print_classification_report(y_true, y_pred):
    print("\nClassification Report:")
    print("-" * 55)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3
    )
    print(report)
    return report


def plot_confusion_matrix(y_true, y_pred):
    print("Saving confusion matrix...")

    try:
        import seaborn as sns
    except ImportError:
        os.system("pip install seaborn")
        import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()

    os.makedirs("training", exist_ok=True)
    plt.savefig("training/confusion_matrix.png", dpi=150)
    plt.close()

    print("✅ Confusion matrix saved to training/confusion_matrix.png")
    print("\nDiagonal = Correct predictions:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {cm[i][i]} correct out of {cm[i].sum()} total")


def save_report(val_loss, val_acc, report):
    path = "training/evaluation_report.txt"
    os.makedirs("training", exist_ok=True)

    with open(path, "w") as f:
        f.write("GEOSPATIAL SURFACE RECOGNITION SYSTEM\n")
        f.write("Evaluation Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Validation Loss:     {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc * 100:.2f}%\n")
        f.write(f"Target Accuracy:     85.00%\n")
        f.write(f"Status: {'PASSED' if val_acc >= 0.85 else 'BELOW TARGET'}\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 45 + "\n")
        f.write(report)

    print(f"\n✅ Report saved to {path}")


if __name__ == "__main__":
    print("=" * 55)
    print("  GEOSPATIAL SURFACE RECOGNITION SYSTEM")
    print("  Step 5: Model Evaluation")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print("   Run step4_train.py first.")
        exit(1)

    print(f"\nLoading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")

    val_ds            = load_val_dataset()
    val_loss, val_acc = evaluate(model, val_ds)
    y_true, y_pred    = get_predictions(model, val_ds)
    report            = print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    save_report(val_loss, val_acc, report)

    print("\n" + "=" * 55)
    print("  EVALUATION COMPLETE ✅")
    print("  Files saved in training/ folder:")
    print("    - confusion_matrix.png")
    print("    - evaluation_report.txt")
    print("=" * 55)