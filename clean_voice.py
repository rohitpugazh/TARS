from pydub import AudioSegment, silence

def remove_silence(input_path, output_path, silence_thresh=-40, min_silence_len=500):
    """
    Removes silent segments from an audio file.
    
    Parameters:
      input_file (str): Path to the input audio file.
      output_file (str): Path to save the output file with silence removed.
      silence_thresh (int): Silence threshold in dBFS (default -40).
      min_silence_len (int): Minimum length for a silence segment (in ms) (default 500).
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_path)
    
    # Detect non-silent parts of the audio
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    if not non_silent_ranges:
        print("No non-silent parts detected in the audio.")
        return

    # Extract non-silent segments and concatenate them
    non_silent_audio = AudioSegment.empty()
    for start, end in non_silent_ranges:
        non_silent_audio += audio[start:end]
    
    # Export the processed audio
    non_silent_audio.export(output_path, format="wav")
    print(f"Silence removed. Output saved to {output_path}")

if __name__ == "__main__":
    input_path = 'clean_bill2.wav'
    output_path = 'final_bill2.wav'
    remove_silence(input_path, output_path)
# Usage: python remove_silence.py input.wav output.wav