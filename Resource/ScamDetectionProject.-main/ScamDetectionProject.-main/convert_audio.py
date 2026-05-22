import os
from pydub import AudioSegment

# Working directory-ஐ set செய்யவும்
os.chdir(r"C:\Users\gnath\OneDrive\Desktop\ScamDetectionProject")

# உண்மையான கோப்புப் பெயர்களை பயன்படுத்தவும்
for i in range(1, 11):  # real1.mp3 முதல் real10.mp3 வரை
    input_file = f"real{i}.mp3"
    output_file = f"real{i}.wav"
    
    if os.path.exists(input_file):
        try:
            sound = AudioSegment.from_mp3(input_file)
            sound.export(output_file, format="wav")
            print(f"{input_file} → {output_file} மாற்றப்பட்டது")
        except Exception as e:
            print(f"பிழை {input_file}: {str(e)}")
    else:
        print(f"{input_file} கிடைக்கவில்லை")