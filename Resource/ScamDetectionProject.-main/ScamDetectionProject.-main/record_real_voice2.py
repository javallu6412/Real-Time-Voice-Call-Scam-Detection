import sounddevice as sd
from scipy.io.wavfile import write
import os

# Ensure folder exists
os.makedirs('voice/real', exist_ok=True)

fs = 16000  # Sampling rate
seconds = 3  # Duration of recording

print("Recording real voice 2...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write('voice/real/real2.wav', fs, recording)
print("Saved real2.wav")
