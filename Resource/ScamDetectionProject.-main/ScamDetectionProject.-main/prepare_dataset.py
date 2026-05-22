# prepare_dataset.py

import os
import pandas as pd

def prepare_dataset(base_dir="new_samples", output_csv="dataset.csv"):
    # Define folder paths
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")

    # Collect real and fake files
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".wav")]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".wav")]

    # Print counts
    print(f"Real files found: {len(real_files)}")
    print(f"Fake files found: {len(fake_files)}")

    # Create dataframe
    data = []
    for f in real_files:
        data.append([f, 0])  # 0 = real
    for f in fake_files:
        data.append([f, 1])  # 1 = fake

    df = pd.DataFrame(data, columns=["filepath", "label"])

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

if __name__ == "__main__":
    prepare_dataset()
