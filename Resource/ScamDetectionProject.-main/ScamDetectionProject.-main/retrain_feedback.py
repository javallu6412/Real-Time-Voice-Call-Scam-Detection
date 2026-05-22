# retrain_feedback.py
import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1️⃣ Extract Features from Feedback Samples
# ---------------------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Paths for feedback folders
real_folder = "new_samples/real"
fake_folder = "new_samples/fake"

X_new = []
y_new = []

# Extract features from REAL files
for file in os.listdir(real_folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(real_folder, file))
        X_new.append(features)
        y_new.append(1)  # 1 = REAL

# Extract features from FAKE files
for file in os.listdir(fake_folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(fake_folder, file))
        X_new.append(features)
        y_new.append(0)  # 0 = FAKE

X_new = np.array(X_new)
y_new = np.array(y_new)

print("✅ Features from feedback samples extracted!")

# ---------------------------
# 2️⃣ Load Old Model & Combine Old + New Data
# ---------------------------
# Load existing model
model = joblib.load("voice_model.pkl")

# Load original training data
X_old = np.load("X_features.npy")
y_old = np.load("y_labels.npy")

# Combine old + new feedback data
X_combined = np.vstack((X_old, X_new))
y_combined = np.hstack((y_old, y_new))

print("✅ Old + new data combined!")

# ---------------------------
# 3️⃣ Retrain the Model
# ---------------------------
model.fit(X_combined, y_combined)

# Save updated model
joblib.dump(model, "voice_model.pkl")
print("✅ Model updated with feedback samples!")

