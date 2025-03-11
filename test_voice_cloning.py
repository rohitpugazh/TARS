import os
import random
from TTS.api import TTS

# Initialize the TTS model (Tacotron2-DDC)
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to("cuda")

# Text to be synthesized
text = "Hello, this is a cloned voice speaking. My name is Bill Irwin and I'm a voice actor."

# Path to the preprocessed dataset
processed_dataset_path = "tts_dataset/wavs"

# Select 10 random reference samples from the dataset
speaker_wavs = [
    os.path.join(processed_dataset_path, f)
    for f in random.sample(os.listdir(processed_dataset_path), 10)
    if f.endswith(".wav")
]

print("Selected reference samples: ", speaker_wavs)

# Perform TTS and save the output
tts.tts_to_file(text=text, speaker_wav=speaker_wavs, file_path="output.wav")

print("TTS synthesis complete. Output saved as 'output.wav'.")