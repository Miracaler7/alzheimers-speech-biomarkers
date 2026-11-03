"""
Module: 01_audio_preprocessing.py
Standardizes Pitt Corpus audio: converts raw audio to clean 16 kHz mono .wav,
normalizes amplitude, and removes edge silence while preserving internal hesitations.
"""
import librosa
import soundfile as sf
from pathlib import Path

def preprocess_audio_pipeline(raw_dir: Path, output_dir: Path, target_sr: int = 16000):
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = list(raw_dir.glob('*.mp3')) + list(raw_dir.glob('*.wav'))
    
    print(f"Found {len(audio_files)} recordings to process.")
    for f in audio_files:
        y, sr = librosa.load(f, sr=target_sr, mono=True)
        # Trim leading and trailing silence (preserve clinical interior pauses)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        # Peak normalize
        y_norm = librosa.util.normalize(y_trimmed)
        
        out_file = output_dir / f"{f.stem}.wav"
        sf.write(out_file, y_norm, target_sr)

if __name__ == '__main__':
    base = Path('/content/drive/MyDrive/mini project 5j')
    preprocess_audio_pipeline(base / 'Audio Spech-PittCorpus', base / 'processed_audio')
