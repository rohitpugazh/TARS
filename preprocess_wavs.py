import os
import librosa
import soundfile as sf

# Define paths
INPUT_FOLDER = "voice_samples"
OUTPUT_FOLDER = "processed_voice_samples"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_audio(file_path, output_path):
    # Load audio
    y, _ = librosa.load(file_path, sr=22050)
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Save processed file
    sf.write(output_path, y_trimmed, 22050)
    print(f"Processed: {output_path}")

# Process all files
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".wav"):
        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, file)
        process_audio(input_path, output_path)

print("Processing complete. Files saved in {OUTPUT_FOLDER}.")
# Output: Processing complete. Files saved in processed_voice_samples.