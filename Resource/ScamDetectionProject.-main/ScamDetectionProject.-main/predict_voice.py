import os
import numpy as np
import joblib
import librosa

# Load trained model
model = joblib.load("voice_model.pkl")

# Function to extract features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Folder paths
real_folder = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\real_wav_aug"
fake_folder = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\fake_wav_aug"

# Predict REAL files
print("\n--- Predictions for REAL files ---")
for file in os.listdir(real_folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(real_folder, file)).reshape(1, -1)
        prediction = model.predict(features)[0]
        print(f"{file}: {'REAL' if prediction==1 else 'FAKE'}")

# Predict FAKE files
print("\n--- Predictions for FAKE files ---")
for file in os.listdir(fake_folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(fake_folder, file)).reshape(1, -1)
        prediction = model.predict(features)[0]
        print(f"{file}: {'REAL' if prediction==1 else 'FAKE'}")
