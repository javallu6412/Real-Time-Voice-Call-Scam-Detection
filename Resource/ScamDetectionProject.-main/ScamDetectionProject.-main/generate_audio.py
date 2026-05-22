import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Save text to audio file (WAV format)
engine.save_to_file("This is a scam call. Please share your OTP.", "test_audio.wav")

# Run the engine
engine.runAndWait()

print("✅ test_audio.wav file created successfully!")
