import tempfile
import whisper
import sounddevice as sd
import numpy as np

from scipy.io import wavfile

class WakeWordHandler:
    def __init__(self):
        # Use the small model for faster wake word detection
        self.whisper_model = whisper.load_model("small")
        self.sample_rate = 22050
        self.is_listening = False
        self.audio_data = []
        self.buffer_duration = 2  # Process 2 seconds of audio at a time

    def start_listening(self):
        """Start listening for wake word"""
        self.is_listening = True
        self.audio_data = []
        try:
            def audio_callback(indata, _, __, status):
                if self.is_listening:
                    self.audio_data.extend(indata.copy())

            # Record audio in chunks
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate):
                print("Listening for 'Hey TARS'...")
                while self.is_listening:
                    sd.sleep(500)  # Check every 0.5 seconds
                    buffer_size = int(self.sample_rate * self.buffer_duration)
                    if len(self.audio_data) >= buffer_size:
                        # Get the last N seconds of audio
                        audio = np.concatenate(self.audio_data[-buffer_size:])
                        # Save temporary audio file
                        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        wavfile.write(temp_file.name, self.sample_rate, audio)
                        # Transcribe with Whisper's tiny model
                        try:
                            result = self.whisper_model.transcribe(
                                temp_file.name,
                                language="en",
                                initial_prompt="hey tars",  # Help bias towards wake word
                                temperature=0
                            )
                            transcribed_text = result["text"].lower().strip()
                            if "hey tars" in transcribed_text:
                                print("Wake word detected!")
                                self.is_listening = False
                                return True
                            # Keep only the last second of audio for overlap
                            self.audio_data = self.audio_data[-int(self.sample_rate):]
                        except Exception as e:
                            print(f"Transcription error: {e}")
                            continue
        except KeyboardInterrupt:
            self.is_listening = False
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False
        return False

    def stop_listening(self):
        """Stop listening for wake word"""
        self.is_listening = False
