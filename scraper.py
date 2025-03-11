import os
import subprocess

def download_audio(video_url, output_folder="voice_samples"):
    """Download audio from a YouTube video in WAV format using yt-dlp."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define output path
    output_path = os.path.join(output_folder, "%(title)s.%(ext)s")

    # yt-dlp command to extract audio in WAV format
    command = [
        "yt-dlp",
        "-x",  # Extract audio only
        "--audio-format", "wav",  # Output format
        "--audio-quality", "0",  # Best quality
        "-o", output_path,
        video_url
    ]

    try:
        print(f"Downloading audio from: {video_url}")
        subprocess.run(command, check=True)
        print(f"Audio downloaded successfully in '{output_folder}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download audio: {e}")

if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    download_audio(video_url)
