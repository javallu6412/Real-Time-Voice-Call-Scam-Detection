import librosa
import os
import numpy as np

voice_dir_real = 'voice/real'
voice_dir_fake = 'voice/fake'

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

mfcc_features = []
labels = []

# Real voices
for file in os.listdir(voice_dir_real):
    path = os.path.join(voice_dir_real, file)
    mfcc_features.append(extract_mfcc(path))
    labels.append(0)  # real

# Fake voices
for file in os.listdir(voice_dir_fake):
    path = os.path.join(voice_dir_fake, file)
    mfcc_features.append(extract_mfcc(path))
    labels.append(1)  # fake

print("MFCC features shape:", np.array(mfcc_features).shape)
print("Labels shape:", np.array(labels).shape)
