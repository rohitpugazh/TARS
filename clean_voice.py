import os
import subprocess
import numpy as np
import librosa
import torch
import soundfile as sf
from pyannote.audio import Pipeline

# Set your Hugging Face token here
HF_TOKEN = "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"

# Step 1: Remove Background Music using Demucs
def remove_bgm(input_audio):
    print("🔊 Removing background music...")
    subprocess.run(["demucs", "--two-stems", "vocals", input_audio], check=True)
    vocals_path = os.path.join("separated", "htdemucs", os.path.splitext(os.path.basename(input_audio))[0], "vocals.wav")
    if not os.path.exists(vocals_path):
        raise FileNotFoundError("Failed to extract vocals. Check Demucs output.")
    print(f"✅ Background music removed. Vocals saved to: {vocals_path}")
    return vocals_path

# Step 2: Perform Speaker Diarization using speaker-diarization-3.1
def diarize_audio(input_audio):
    print("🗣️ Performing speaker diarization using PyAnnote 3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    ).to(torch.device("cuda"))

    diarization = pipeline(
        input_audio,
        max_speakers=2         # Limit to 2 speakers (optional)
    )
    return diarization

# Step 3: Extract Audio for One Speaker
def extract_speaker_audio(input_audio, diarization, target_speaker, output_path):
    print(f"🎙️ Extracting audio for {target_speaker}...")

    # Load the original audio
    audio, sr = librosa.load(input_audio, sr=22050, mono=True)

    # Extract only the segments for the desired speaker
    speaker_audio = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{segment.start:.2f} - {segment.end:.2f}: {speaker}")
        if speaker == target_speaker:
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            speaker_audio.append(audio[start_sample:end_sample])

    # Concatenate all segments
    if speaker_audio:
        final_audio = np.concatenate(speaker_audio)
        sf.write(output_path, final_audio, sr)
        print(f"✅ Extracted speaker audio saved to: {output_path}")
    else:
        print(f"⚠️ No segments found for {target_speaker}.")

# Main Function
def main(input_audio, target_speaker="Speaker_1", output_audio="speaker_output.wav"):
    # Step 1: Remove Background Music
    vocals_path = remove_bgm(input_audio)

    # Step 2: Perform Speaker Diarization
    diarization = diarize_audio(vocals_path)

    # Step 3: Extract One Speaker
    extract_speaker_audio(vocals_path, diarization, target_speaker, output_audio)

if __name__ == "__main__":
    INPUT_AUDIO = "voice_samples/an-interview-with-bill-irwinthe-directors-cut.wav"
    TARGET_SPEAKER = "SPEAKER_01"
    OUTPUT_AUDIO = "extracted_speaker.wav"

    main(INPUT_AUDIO, TARGET_SPEAKER, OUTPUT_AUDIO)
