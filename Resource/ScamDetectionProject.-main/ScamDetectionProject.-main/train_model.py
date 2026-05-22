import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Step 1: Load dataset ---
df = pd.read_csv("dataset.csv")
print("📂 Dataset columns:", df.columns)
print(df.head())

# --- Step 2: Feature Extraction from audio ---
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("❌ Error extracting:", file_path, e)
        return np.zeros(13)  # fallback features

features = []
labels = []

for _, row in df.iterrows():
    feat = extract_features(row["filepath"])
    features.append(feat)
    labels.append(row["label"])

X = np.array(features)
y = np.array(labels)

# --- Step 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 5: Evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# --- Step 6: Save model ---
joblib.dump(model, "voice_model.pkl")
print("✅ Voice Model saved as voice_model.pkl")

