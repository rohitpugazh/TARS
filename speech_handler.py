import os
import logging
import warnings
import tempfile
import whisper
import torch
from TTS.api import TTS

# Configure logging
logging.getLogger("TTS").setLevel(logging.ERROR)
logging.getLogger("whisper").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

class SpeechHandler:
    def __init__(self, ref_audio_path=None):
        try:
            # Initialize Whisper model
            self.whisper_model = whisper.load_model("base")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize TTS with custom finetuned XTTS2 model
            # print("Initializing TARS TTS model...")
            model_dir = os.path.join("models", "TARS")
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"TARS model directory not found at {model_dir}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"TARS config not found at {config_path}")
            self.tts = TTS(model_path=model_dir, config_path=config_path, progress_bar=False).to(self.device)

            # Set reference audio
            self.ref_audio_path = ref_audio_path
            if ref_audio_path and not os.path.exists(ref_audio_path):
                print(f"Warning: Reference audio file not found at {ref_audio_path}")
                self.ref_audio_path = None

            # print(f"Using device: {self.device}")
            # if self.ref_audio_path:
            #     print(f"Using voice from: {self.ref_audio_path}")
        except Exception as e:
            print(f"Error initializing speech models: {e}")
            raise

    def transcribe_audio(self, audio_file):
        """Convert speech to text using Whisper"""
        if not os.path.exists(audio_file):
            print("Audio file not found")
            return None
        try:
            result = self.whisper_model.transcribe(audio_file)
            return result["text"].strip()
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def generate_speech(self, text):
        """Convert text to speech using TARS TTS"""
        if not text:
            print("No text provided for speech generation")
            return None

        try:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "tts_output.wav")

            # Generate speech using TARS TTS
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=self.ref_audio_path,  # Use reference audio for voice cloning
                language="en",
                speed=1.2
            )

            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            print(f"Error during speech generation: {e}")
            return None
