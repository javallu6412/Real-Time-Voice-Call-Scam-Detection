from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load model
model = joblib.load("voice_model.pkl")
print("✅ Voice model loaded successfully!")

# Feature extraction
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("❌ Error extracting:", file_path, e)
        return np.zeros(13)

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    predictions = []
    chart_file = None

    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return render_template("index.html", predictions=["No files selected!"])

        real_count = 0
        fake_count = 0

        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            features = extract_features(file_path).reshape(1, -1)
            pred = model.predict(features)[0]
            result = "REAL" if pred == 0 else "FAKE"

            predictions.append(f"{file.filename} --> {result}")

            if pred == 0:
                real_count += 1
            else:
                fake_count += 1

        # Plot bar chart
        plt.figure(figsize=(4,4))
        plt.bar(["REAL","FAKE"], [real_count,fake_count], color=["green","red"])
        plt.title("Prediction Summary")
        plt.ylabel("Count")
        chart_file = os.path.join(STATIC_FOLDER, "chart.png")
        plt.savefig(chart_file)
        plt.close()

        # Provide relative path for HTML
        chart_file = "static/chart.png"

    return render_template("index.html", predictions=predictions, chart_file=chart_file)

if __name__ == "__main__":
    app.run(debug=True)
