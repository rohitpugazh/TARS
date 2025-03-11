from pydub import AudioSegment

def trim_audio(input_path, output_path):
    """Cut the first 3 seconds from an audio file."""
    # Load the audio file
    audio = AudioSegment.from_file(input_path)

    # Cut the first 3 seconds
    trimmed_audio = audio[500:]

    # Save the trimmed audio
    trimmed_audio.export(output_path, format="wav")
    print(f"✅ Trimmed audio saved to: {output_path}")

# Example usage
input_file = "extracted_speaker.wav"
output_file = "extracted_speaker.wav"
trim_audio(input_file, output_file)
