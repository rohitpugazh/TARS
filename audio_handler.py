
import os
import tempfile
import time
import sounddevice as sd
import numpy as np
from scipy.io import wavfile

class AudioHandler:
    def __init__(self, sample_rate=22050, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_data = []
        self.is_recording = False
        self.silence_threshold = 0.015  # Increased threshold for better noise rejection
        self.silence_duration = 1.2  # Keep the same duration
        self.last_sound_time = None
        self.min_record_time = 0.3  # Keep the same minimum time
        self.is_speaking = False  # Track if we're currently speaking

    def is_silent(self, indata):
        """Check if the audio chunk is silent"""
        return np.max(np.abs(indata)) < self.silence_threshold

    def record_audio(self):
        """Record audio from microphone with automatic silence detection"""
        if self.is_speaking:
            print("Processing previous response...")
            return None

        print("Listening... (speak now)")
        self.audio_data = []
        self.is_recording = True
        self.last_sound_time = time.time()
        start_time = time.time()

        try:
            def callback(indata, _, __, status):
                if status:
                    print(f"Audio callback status: {status}")
                if self.is_recording:
                    self.audio_data.extend(indata.copy())
                    current_time = time.time()

                    # Update last_sound_time if sound is detected
                    if not self.is_silent(indata):
                        self.last_sound_time = current_time

                    # Check if we've had enough silence and minimum duration has passed
                    elapsed_time = current_time - start_time
                    silence_time = current_time - self.last_sound_time

                    if (elapsed_time > self.min_record_time and
                        silence_time >= self.silence_duration):
                        self.is_recording = False
                        print("\nProcessing...")  # Indicate we're done listening

            with sd.InputStream(callback=callback,
                              channels=self.channels,
                              samplerate=self.sample_rate):
                while self.is_recording:
                    sd.sleep(100)

        except KeyboardInterrupt:
            self.is_recording = False
        except Exception as e:
            print(f"Error during recording: {e}")
            return None

        if len(self.audio_data) == 0:
            return None

        return np.concatenate(self.audio_data)

    def save_audio(self, audio_data, filename):
        """Save audio data to a WAV file and return the audio data"""
        try:
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                # If audio_data is a file path, read it
                _, data = wavfile.read(audio_data)
                return data
            elif isinstance(audio_data, np.ndarray):
                # If audio_data is numpy array, save it and return it
                wavfile.write(filename, self.sample_rate, audio_data)
                return audio_data
            else:
                print("Invalid audio data format")
                return None
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def play_audio(self, audio_data):
        """Play audio data through speakers"""
        if audio_data is None:
            return
        try:
            self.is_speaking = True
            sd.play(audio_data, self.sample_rate)
            sd.wait()
            self.is_speaking = False
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_speaking = False

    def save_temp_wav(self, audio_data):
        """Save audio data to a temporary WAV file and return the path"""
        if audio_data is None:
            return None
        try:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "temp_audio.wav")
            wavfile.write(temp_path, self.sample_rate, audio_data)
            return temp_path
        except Exception as e:
            print(f"Error saving temporary file: {e}")
            return None
