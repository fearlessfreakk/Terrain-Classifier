"""
STEP 3: Build CNN Model
────────────────────────
Defines and compiles the CNN architecture.
Prints model summary showing all layers and parameter counts.

Usage:
    python training/step3_build_model.py
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, LeakyReLU
)

# ── Configuration ───────────────────────────────────────────
IMG_HEIGHT = 150
IMG_WIDTH  = 150


def build_model():
    print("=" * 50)
    print("  STEP 3: Building CNN Model")
    print("=" * 50)

    model = Sequential([

        # Block 1 — basic features (edges, colors)
        # Input: 150×150×3 → Output: 148×148×64
        Conv2D(64, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="conv2d_block1"),
        LeakyReLU(negative_slope=0.01, name="leaky_relu_1"),
        MaxPooling2D(2, 2, name="maxpool_1"),
        # After MaxPool: 74×74×64

        # Block 2 — intermediate features (textures)
        # Input: 74×74×64 → Output: 72×72×128
        Conv2D(128, (3, 3), name="conv2d_block2"),
        LeakyReLU(negative_slope=0.01, name="leaky_relu_2"),
        MaxPooling2D(2, 2, name="maxpool_2"),
        # After MaxPool: 36×36×128

        # Block 3 — high-level features (terrain patterns)
        # Input: 36×36×128 → Output: 34×34×256
        Conv2D(256, (3, 3), name="conv2d_block3"),
        LeakyReLU(negative_slope=0.01, name="leaky_relu_3"),
        MaxPooling2D(2, 2, name="maxpool_3"),
        # After MaxPool: 17×17×256

        # Classifier head
        Flatten(name="flatten"),
        # 17×17×256 = 73,984 values

        Dense(512, name="dense_512"),
        LeakyReLU(negative_slope=0.01, name="leaky_relu_4"),
        Dropout(0.5, name="dropout"),

        Dense(4, name="output")
        # 4 neurons — Grassy=0, Marshy=1, Rocky=2, Sandy=3

    ], name="terrain_classifier")

    return model


def compile_model(model):
    print("\nCompiling model...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("✅ Model compiled.")
    return model


def print_summary(model):
    print("\nModel Architecture Summary:")
    print("-" * 50)
    model.summary()

    total_params = model.count_params()
    size_mb      = (total_params * 4) / (1024 * 1024)

    print(f"\nTotal Parameters:  {total_params:,}")
    print(f"Estimated Size:    {size_mb:.1f} MB (weights only)")
    print(f"Input Shape:       {IMG_HEIGHT}×{IMG_WIDTH}×3")
    print(f"Output Classes:    4 (Grassy, Marshy, Rocky, Sandy)")


if __name__ == "__main__":
    model = build_model()
    model = compile_model(model)
    print_summary(model)

    print("\n✅ Step 3 complete. Ready for Step 4.")
    print("=" * 50)