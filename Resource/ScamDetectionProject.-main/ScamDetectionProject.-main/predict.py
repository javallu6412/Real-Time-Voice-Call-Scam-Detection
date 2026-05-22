import joblib
import librosa
import numpy as np
import sys

# --- Load model ---
model = joblib.load("voice_model.pkl")

# --- Feature extractor ---
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("❌ Error extracting:", file_path, e)
        return np.zeros(13)

# --- Take input file from command line ---
if len(sys.argv) < 2:
    print("⚠️ Usage: python predict.py path/to/file.wav")
    sys.exit()

file_path = sys.argv[1]

# --- Extract features & predict ---
features = extract_features(file_path).reshape(1, -1)
prediction = model.predict(features)[0]

result = "Real" if prediction == 0 else "Fake"
print(f"🎤 File: {file_path} --> Prediction: {result}")


