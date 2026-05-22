# realtime_scam_detector.py
"""
Real-time scam detection simulator for pre-recorded audio files (.wav)
Processes audio in chunks to simulate real-time streaming behavior
"""

import argparse
import time
import os
import threading
import queue
from collections import deque
import numpy as np
import whisper
import joblib
import torch
import librosa
from transformers import DistilBertTokenizer, DistilBertModel

class AudioFileScamDetector:
    def __init__(self, model_dir, whisper_size="tiny", chunk_duration=3.0, 
                 hop_duration=1.5, context_chunks=3, alert_threshold=0.6, 
                 simulate_realtime=True):
        
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.context_chunks = context_chunks
        self.alert_threshold = alert_threshold
        self.simulate_realtime = simulate_realtime
        
        # Audio settings
        self.sample_rate = 16000  # Whisper's preferred sample rate
        
        # Queues for threading
        self.audio_queue = queue.Queue(maxsize=20)
        self.text_queue = queue.Queue(maxsize=20)
        self.running = False
        
        # Context storage
        self.recent_texts = deque(maxlen=context_chunks)
        self.detection_results = []
        
        # Load models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models(model_dir, whisper_size)
        
    def load_models(self, model_dir, whisper_size):
        """Load all required models"""
        print("Loading models...")
        
        # Load classifier
        self.clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        
        # Load embedder
        with open(os.path.join(model_dir, "embedder_model.txt"), "r") as f:
            embedder_name = f.read().strip()
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(embedder_name)
        self.bert_model = DistilBertModel.from_pretrained(embedder_name).to(self.device)
        
        # Load Whisper
        self.whisper_model = whisper.load_model(whisper_size)
        
        print(f"Models loaded successfully on {self.device}")
    
    def load_audio_file(self, audio_path):
        """Load audio file and resample to 16kHz"""
        print(f"Loading audio file: {audio_path}")
        
        # Load audio with librosa (handles various formats)
        audio_data, original_sr = librosa.load(audio_path, sr=None)
        
        # Resample to 16kHz if needed
        if original_sr != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            print(f"Resampled from {original_sr}Hz to {self.sample_rate}Hz")
        
        duration = len(audio_data) / self.sample_rate
        print(f"Audio loaded: {duration:.2f} seconds")
        
        return audio_data
    
    def chunk_audio(self, audio_data):
        """Generate audio chunks with overlap"""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        hop_samples = int(self.hop_duration * self.sample_rate)
        
        chunks = []
        for start in range(0, len(audio_data) - chunk_samples + 1, hop_samples):
            end = start + chunk_samples
            chunk = audio_data[start:end]
            timestamp = start / self.sample_rate
            chunks.append((chunk, timestamp))
        
        # Add final chunk if there's remaining audio
        if len(audio_data) % hop_samples != 0:
            final_chunk = audio_data[-chunk_samples:]
            final_timestamp = (len(audio_data) - chunk_samples) / self.sample_rate
            if final_timestamp >= 0:
                chunks.append((final_chunk, final_timestamp))
        
        return chunks
    
    def audio_streaming_thread(self, audio_chunks):
        """Simulate real-time audio streaming by feeding chunks with delays"""
        print(f"Starting audio streaming thread with {len(audio_chunks)} chunks")
        
        for i, (chunk, timestamp) in enumerate(audio_chunks):
            if not self.running:
                break
                
            try:
                self.audio_queue.put((chunk, timestamp, i), timeout=1.0)
                print(f"Streamed chunk {i+1}/{len(audio_chunks)} at {timestamp:.1f}s")
                
                # Simulate real-time delay
                if self.simulate_realtime and i < len(audio_chunks) - 1:
                    time.sleep(self.hop_duration)
                    
            except queue.Full:
                print("Audio queue full, dropping chunk")
        
        print("Audio streaming completed")
    
    def transcription_thread(self):
        """Thread for transcribing audio chunks"""
        print("Starting transcription thread")
        
        while self.running:
            try:
                chunk_data = self.audio_queue.get(timeout=2.0)
                if chunk_data is None:  # Sentinel value to stop
                    break
                    
                audio_chunk, timestamp, chunk_id = chunk_data
                
                # Transcribe with Whisper
                start_time = time.time()
                result = self.whisper_model.transcribe(
                    audio_chunk, 
                    fp16=False,
                    language="en",
                    word_timestamps=False
                )
                transcription_time = time.time() - start_time
                
                text = result.get("text", "").strip()
                
                if text:
                    print(f"[{timestamp:.1f}s] Transcribed in {transcription_time:.2f}s: {text}")
                    try:
                        self.text_queue.put((text, timestamp, chunk_id), timeout=1.0)
                    except queue.Full:
                        print("Text queue full, dropping transcription")
                else:
                    print(f"[{timestamp:.1f}s] No speech detected")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
        
        print("Transcription thread finished")
    
    def embed_text(self, text):
        """Generate DistilBERT embedding for text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        return outputs.last_hidden_state[:,0,:].cpu().numpy()[0]
    
    def classification_thread(self):
        """Thread for scam classification"""
        print("Starting classification thread")
        
        while self.running:
            try:
                text_data = self.text_queue.get(timeout=2.0)
                if text_data is None:  # Sentinel value to stop
                    break
                    
                text, timestamp, chunk_id = text_data
                
                # Add to context
                self.recent_texts.append(text)
                
                # Create context from recent texts
                context = " ".join(self.recent_texts)
                
                # Generate embedding and classify
                start_time = time.time()
                embedding = self.embed_text(context)
                embedding = embedding.reshape(1, -1)
                
                scam_prob = self.clf.predict_proba(embedding)[0][1]
                classification_time = time.time() - start_time
                
                # Store result
                result = {
                    'timestamp': timestamp,
                    'chunk_id': chunk_id,
                    'text': text,
                    'context': context,
                    'scam_probability': scam_prob,
                    'classification_time': classification_time,
                    'is_scam_alert': scam_prob >= self.alert_threshold
                }
                self.detection_results.append(result)
                
                print(f"[{timestamp:.1f}s] Scam probability: {scam_prob:.3f} (classified in {classification_time:.3f}s)")
                
                if scam_prob >= self.alert_threshold:
                    print(f"🚨 SCAM ALERT! 🚨 (Confidence: {scam_prob:.1%})")
                    self.on_scam_detected(scam_prob, context, timestamp)
                
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and chunk_id % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Classification error: {e}")
        
        print("Classification thread finished")
    
    def on_scam_detected(self, probability, context, timestamp):
        """Handle scam detection event"""
        print(f"⚠️  SCAM DETECTED at {timestamp:.1f}s with {probability:.1%} confidence")
        print(f"Context: {context[:200]}...")
        
        # Add your custom actions here:
        # - Log to file
        # - Send alert
        # - Mark timestamp for review
        # - etc.
    
    def process_audio_file(self, audio_path):
        """Process a single audio file for scam detection"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        
        # Load and chunk audio
        audio_data = self.load_audio_file(audio_path)
        audio_chunks = self.chunk_audio(audio_data)
        
        print(f"Created {len(audio_chunks)} chunks of {self.chunk_duration}s each")
        print(f"Overlap: {self.chunk_duration - self.hop_duration}s")
        print(f"Real-time simulation: {'ON' if self.simulate_realtime else 'OFF'}")
        
        # Clear previous results
        self.detection_results = []
        self.recent_texts.clear()
        
        # Start processing
        self.running = True
        
        # Start threads
        threads = [
            threading.Thread(target=self.audio_streaming_thread, args=(audio_chunks,), daemon=True),
            threading.Thread(target=self.transcription_thread, daemon=True),
            threading.Thread(target=self.classification_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for audio streaming to complete
        threads[0].join()
        
        # Wait a bit for remaining processing
        time.sleep(3)
        
        # Signal other threads to stop
        self.audio_queue.put(None)
        self.text_queue.put(None)
        self.running = False
        
        # Wait for threads to finish
        for thread in threads[1:]:
            thread.join(timeout=5)
        
        return self.detection_results
    
    def print_summary(self, results):
        """Print detection summary"""
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*60}")
        print("DETECTION SUMMARY")
        print(f"{'='*60}")
        
        total_chunks = len(results)
        scam_alerts = sum(1 for r in results if r['is_scam_alert'])
        max_prob = max(r['scam_probability'] for r in results)
        avg_prob = sum(r['scam_probability'] for r in results) / total_chunks
        
        print(f"Total chunks processed: {total_chunks}")
        print(f"Scam alerts triggered: {scam_alerts}")
        print(f"Maximum scam probability: {max_prob:.3f}")
        print(f"Average scam probability: {avg_prob:.3f}")
        
        if scam_alerts > 0:
            print(f"\nScam alert timestamps:")
            for result in results:
                if result['is_scam_alert']:
                    print(f"  {result['timestamp']:.1f}s - {result['scam_probability']:.3f}")
        
        print(f"\nFull transcript:")
        for result in results:
            if result['text']:
                print(f"  [{result['timestamp']:.1f}s] {result['text']}")

def main():
    parser = argparse.ArgumentParser(description="Real-time scam detection for audio files")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model_dir", default="models", help="Directory with trained models")
    parser.add_argument("--whisper_size", default="tiny", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model size")
    parser.add_argument("--chunk_duration", type=float, default=3.0, 
                       help="Audio chunk duration in seconds")
    parser.add_argument("--hop_duration", type=float, default=1.5, 
                       help="Time between chunks in seconds") 
    parser.add_argument("--context_chunks", type=int, default=3,
                       help="Number of recent chunks to use for context")
    parser.add_argument("--alert_threshold", type=float, default=0.6,
                       help="Probability threshold for scam alerts")
    parser.add_argument("--no_realtime_simulation", action="store_true",
                       help="Process chunks as fast as possible (no delays)")
    parser.add_argument("--save_results", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    detector = AudioFileScamDetector(
        model_dir=args.model_dir,
        whisper_size=args.whisper_size,
        chunk_duration=args.chunk_duration,
        hop_duration=args.hop_duration,
        context_chunks=args.context_chunks,
        alert_threshold=args.alert_threshold,
        simulate_realtime=not args.no_realtime_simulation
    )
    
    # Process the audio file
    start_time = time.time()
    results = detector.process_audio_file(args.audio)
    total_time = time.time() - start_time
    
    # Print summary
    detector.print_summary(results)
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.save_results}")

if __name__ == "__main__":
    main()