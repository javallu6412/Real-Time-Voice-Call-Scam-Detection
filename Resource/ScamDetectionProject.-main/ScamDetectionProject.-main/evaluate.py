import os
import numpy as np
import librosa
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -----------------------------
# Load trained voice model
# -----------------------------
model = joblib.load("voice_model.pkl")
print("✅ Voice model loaded successfully!")

# -----------------------------
# Function to extract MFCC features
# -----------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("❌ Error extracting:", file_path, e)
        return np.zeros(13)

# -----------------------------
# Paths to test dataset folders
# -----------------------------
real_folder = "test_audios/real"
fake_folder = "test_audios/fake"

# -----------------------------
# Collect files
# -----------------------------
real_files = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith(".wav")]
fake_files = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith(".wav")]

# -----------------------------
# Prepare features and labels
# -----------------------------
X = []
y_true = []

for f in real_files:
    X.append(extract_features(f))
    y_true.append(0)  # Real = 0

for f in fake_files:
    X.append(extract_features(f))
    y_true.append(1)  # Fake = 1

X = np.array(X)
y_true = np.array(y_true)

# -----------------------------
# Make predictions
# -----------------------------
y_pred = model.predict(X)

# -----------------------------
# Show metrics
# -----------------------------
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# -----------------------------
# Plot REAL vs FAKE predictions
# -----------------------------
unique, counts = np.unique(y_pred, return_counts=True)
plt.bar(["Real", "Fake"], counts, color=["green", "red"])
plt.title("Predicted REAL vs FAKE Audio Counts")
plt.ylabel("Number of files")
plt.show()
