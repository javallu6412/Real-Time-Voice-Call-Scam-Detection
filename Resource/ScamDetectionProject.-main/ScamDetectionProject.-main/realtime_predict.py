import sounddevice as sd       # For real-time audio recording
import numpy as np
import librosa                 # For audio processing
import joblib                  # For loading the trained model
# Load trained voice model
model = joblib.load("voice_model.pkl")
print("Voice model loaded successfully!")
def record_audio(duration=3, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait until recording is finished
    audio = audio.flatten()
    print("Recording complete!")
    return audio
def extract_features(audio, sr=16000):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean
# Record audio
audio = record_audio(duration=3)

# Extract features
features = extract_features(audio).reshape(1, -1)

# Predict
prediction = model.predict(features)

# Output result
if prediction[0] == 1:
    print("Prediction: REAL")
else:
    print("Prediction: FAKE")
while True:
    audio = record_audio(duration=3)
    features = extract_features(audio).reshape(1, -1)
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        print("Prediction: REAL")
    else:
        print("Prediction: FAKE")
    
    cont = input("Record another sample? (y/n): ")
    if cont.lower() != 'y':
        break

from feedback import save_feedback
prediction = model.predict(features)[0]
if prediction == 1:
    predicted_label = "REAL"
else:
    predicted_label = "FAKE"

print(f"Prediction: {predicted_label}")

# Ask user for feedback
user_label = input("Is this correct? If not, enter correct label (REAL/FAKE) or press Enter to skip: ").strip()
if user_label:
    save_feedback("temp.wav", user_label)

