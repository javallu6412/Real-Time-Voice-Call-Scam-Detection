from pydub import AudioSegment
import os
import random

# Input folders (original wav files)
real_input = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\real_wav"
fake_input = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\fake_wav"

# Output folders (augmented wav files)
real_aug_output = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\real_wav_aug"
fake_aug_output = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\fake_wav_aug"

os.makedirs(real_aug_output, exist_ok=True)
os.makedirs(fake_aug_output, exist_ok=True)

def augment_audio(file_path, output_folder, idx):
    sound = AudioSegment.from_wav(file_path)
    
    # Random speed change
    speed = random.uniform(0.9, 1.1)
    sound = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    }).set_frame_rate(sound.frame_rate)
    
    # Random volume change
    change_db = random.uniform(-5, 5)
    sound = sound + change_db
    
    # Save augmented file
    out_path = os.path.join(output_folder, f"aug_{idx}.wav")
    sound.export(out_path, format="wav")
    print(f"Saved: {out_path}")

# Augment real files
idx = 1
for file in os.listdir(real_input):
    if file.endswith(".wav"):
        for i in range(10):
            augment_audio(os.path.join(real_input, file), real_aug_output, idx)
            idx += 1

# Augment fake files
idx = 1
for file in os.listdir(fake_input):
    if file.endswith(".wav"):
        for i in range(10):
            augment_audio(os.path.join(fake_input, file), fake_aug_output, idx)
            idx += 1

