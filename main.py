"""Module to transcribe audio files using the Whisper library."""

import os
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"  # Add FFmpeg to the system path
import torch
import whisper # Import the Whisper library

class SpeechRecognizer:
    """Speech recognition class using the Whisper library."""

    def __init__(self, model_size='small'):
        """Initialize the speech recognizer."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.model = whisper.load_model(model_size).to(device)

    def transcribe(self, audio_file) -> str:
        """Transcribe the given audio file."""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        print("Transcribing audio...")
        result = self.model.transcribe(audio_file)
        return result['text']

if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    INPUT_AUDIO = 'input.wav'  # Provide an existing voice recording

    text = recognizer.transcribe(INPUT_AUDIO)
    print("Recognized text:", text)
