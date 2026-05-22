import os
import numpy as np
import librosa

# Folder paths
real_folder = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\real_wav_aug"
fake_folder = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\fake_wav_aug"

# Lists to store features and labels
X = []  # features
y = []  # labels: 1=real, 0=fake

# Function to extract MFCC features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)      # wav file read pannitu
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # 13 MFCC features
    mfccs_mean = np.mean(mfccs.T, axis=0)            # average per feature
    return mfccs_mean

# Real files → label 1
for file in os.listdir(real_folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(real_folder, file))
        X.append(features)
        y.append(1)

# Fake files → label 0
for file in os.listdir(fake_folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(fake_folder, file))
        X.append(features)
        y.append(0)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Save features & labels
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("Feature extraction complete!")
print("X_features.npy & y_labels.npy saved in current folder.")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

