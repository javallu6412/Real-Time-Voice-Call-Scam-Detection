# real_time_detector.py
"""
Simulated real-time scam detection:
- Splits audio into chunks
- Transcribes with Whisper (local)
- Embeds text with DistilBERT
- Classifies with Logistic Regression
"""

import argparse, time, os
import whisper
import joblib
import numpy as np
from pydub import AudioSegment
from transformers import DistilBertTokenizer, DistilBertModel
import torch

def load_models(model_dir, device):
    clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    with open(os.path.join(model_dir, "embedder_model.txt"), "r") as f:
        embedder_name = f.read().strip()
    tokenizer = DistilBertTokenizer.from_pretrained(embedder_name)
    model = DistilBertModel.from_pretrained(embedder_name).to(device)
    return clf, tokenizer, model

def embed_text(texts, tokenizer, model, device):
    embeddings = []
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:,0,:].cpu().numpy()[0])
    return np.array(embeddings)

def chunk_audio(audio, chunk_ms, hop_ms):
    length = len(audio)
    for start in range(0, max(1, length - chunk_ms + 1), hop_ms):
        yield audio[start:start+chunk_ms], start

def transcribe_chunk(model, audio_chunk, tmp_wav="tmp_chunk.wav"):
    audio_chunk.export(tmp_wav, format="wav")
    result = model.transcribe(tmp_wav, fp16=False)
    return result.get("text","").strip()

def main(args):
    audio = AudioSegment.from_file(args.audio)
    print(f"Loaded audio: {len(audio)/1000:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf, tokenizer, bert_model = load_models(args.model_dir, device)
    whisper_model = whisper.load_model(args.whisper_size)

    chunk_ms, hop_ms = int(args.chunk_sec*1000), int(args.hop_sec*1000)
    recent_texts = []
    threshold = args.alert_threshold

    for chunk, start in chunk_audio(audio, chunk_ms, hop_ms):
        if args.simulate_realtime:
            time.sleep(args.chunk_sec)

        text = transcribe_chunk(whisper_model, chunk)
        print(f"[{start/1000:.1f}s] {text}")

        if not text: 
            continue
        recent_texts.append(text)
        if len(recent_texts) > args.context_chunks:
            recent_texts.pop(0)

        context = " ".join(recent_texts)
        emb = embed_text([context], tokenizer, bert_model, device)
        prob = clf.predict_proba(emb)[0][1]  # scam prob
        print(f"   Scam probability: {prob:.3f}")

        if prob >= threshold:
            print("🚨 ALERT: Scam likely!")
            if args.stop_on_alert:
                break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Audio file path")
    p.add_argument("--model_dir", default="models", help="Trained classifier dir")
    p.add_argument("--chunk_sec", type=float, default=5.0)
    p.add_argument("--hop_sec", type=float, default=2.5)
    p.add_argument("--context_chunks", type=int, default=3)
    p.add_argument("--alert_threshold", type=float, default=0.6)
    p.add_argument("--stop_on_alert", action="store_true")
    p.add_argument("--simulate_realtime", action="store_true")
    p.add_argument("--whisper_size", default="tiny")
    args = p.parse_args()
    main(args)
