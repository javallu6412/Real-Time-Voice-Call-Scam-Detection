import os
import glob

# Folder paths
real_folder = "new_samples/real"
fake_folder = "new_samples/fake"

# Get all .wav files from both folders
real_files = glob.glob(os.path.join(real_folder, "*.wav"))
fake_files = glob.glob(os.path.join(fake_folder, "*.wav"))

print("Real files found:", real_files)
print("Fake files found:", fake_files)

