from pydub import AudioSegment
import os

# Input folders
real_input = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\real"
fake_input = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\fake"

# Output folders
real_output = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\real_wav"
fake_output = r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject\voice\fake_wav"

# Create output folders if not exist
os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

# Function to convert mp3 → wav
def convert_mp3_to_wav(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            mp3_path = os.path.join(input_folder, file)
            wav_path = os.path.join(output_folder, file.replace(".mp3", ".wav"))
            AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
            print(f"Saved: {wav_path}")

# Convert both real & fake
convert_mp3_to_wav(real_input, real_output)
convert_mp3_to_wav(fake_input, fake_output)






