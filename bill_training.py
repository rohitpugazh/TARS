import os
import whisper
from pydub import AudioSegment
import pandas as pd

# Paths
INPUT_AUDIO = "final_bill2.wav"  # Path to cleaned voice clip
OUTPUT_FOLDER = "tts_dataset"
METADATA_FILE = os.path.join(OUTPUT_FOLDER, "metadata.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load audio file
audio = AudioSegment.from_wav(INPUT_AUDIO)

# Split into 5-10 second chunks
segment_length = 7000  # Adjust based on preference (in milliseconds)
segments = [audio[i : i + segment_length] for i in range(0, len(audio), segment_length)]

# Load Whisper model
model = whisper.load_model("small")  # Change to "medium" or "large" if needed

# Load existing metadata if available
if os.path.exists(METADATA_FILE):
    df = pd.read_csv(METADATA_FILE, sep="|", names=["file_path", "transcription"], header=None)
else:
    df = pd.DataFrame(columns=["file_path", "transcription"])

# Transcribe each segment
metadata = []
existing_files = df["file_path"].tolist()
for i, segment in enumerate(segments):
    segment_path = os.path.join(OUTPUT_FOLDER, f"clip_{i}.wav")
    if segment_path in existing_files:
        continue  # Skip existing files

    segment.export(segment_path, format="wav")

    # Transcribe with Whisper
    result = model.transcribe(segment_path)
    transcription = result["text"]

    # Save transcription
    metadata.append([segment_path, transcription])
    print(f"Processed clip {i}: {transcription}")

# Append new data to existing metadata
df = pd.concat([df, pd.DataFrame(metadata, columns=["file_path", "transcription"])], ignore_index=True)
df.to_csv(METADATA_FILE, index=False, sep="|")

print(f"\nDataset ready! {len(segments)} audio clips saved in {OUTPUT_FOLDER}")
print(f"Metadata saved in {METADATA_FILE}")
