"""
STEP 1: Verify Dataset
──────────────────────
Run this FIRST before training.
Checks that dataset folder exists and counts images per class.

Usage:
    python training/step1_verify_dataset.py
"""

import os

DATASET_DIR = "dataset"
EXPECTED_CLASSES = ["Grassy", "Marshy", "Rocky", "Sandy"]

def verify_dataset():
    print("=" * 50)
    print("  STEP 1: Dataset Verification")
    print("=" * 50)

    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ ERROR: '{DATASET_DIR}' folder not found!")
        print("   Make sure you have extracted the Kaggle dataset")
        print("   into a folder called 'dataset' in the project root.")
        return False

    print(f"\n✅ Dataset folder found: {DATASET_DIR}/\n")

    total = 0
    found_classes = []

    for folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(folder_path):
            count = len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            print(f"  {folder:<10}: {count} images")
            total += count
            found_classes.append(folder)

    print(f"\n  {'Total':<10}: {total} images")
    print(f"  Classes found: {found_classes}")

    if sorted(found_classes) == sorted(EXPECTED_CLASSES):
        print("\n✅ All 4 classes verified: Grassy, Marshy, Rocky, Sandy")
    else:
        print(f"\n⚠️  WARNING: Expected {EXPECTED_CLASSES}")
        print(f"             Found    {found_classes}")

    print("\n✅ Dataset verification complete. Ready for Step 2.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    verify_dataset()