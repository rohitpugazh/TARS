# TARS AI Assistant

A TARS-inspired conversational AI assistant that processes voice input, generates responses using Mistral-7B, and replies with synthesized speech.

## Features
- Voice input processing using microphone
- Speech-to-text using OpenAI Whisper
- Text generation using Mistral-7B LLM
- Text-to-speech synthesis with voice cloning
- Interactive conversation loop with wake word detection
- Sarcasm toggle for TARS's personality

## Requirements
- Windows 10/11
- Python 3.11.9 or higher
- CUDA-capable GPU (recommended)
- Required packages (see requirements.txt)

## Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/TARS.git
cd TARS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
   - Mistral-7B Model:
     - Download `mistral-7b-instruct-v0.2.Q6_K.gguf` from [TheBloke's HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
     - Place it in the `models/LLM` directory
   - TTS Model:
     - Download the XTTS2 model files
     - Place `model.pth` in the `models/TTS` directory
     - The `config.json` and training audio files are included in the repository
   - Reference Audio:
     - Place your reference audio file (e.g., reference.wav) in the project root
     - This should be a clear, high-quality recording of the voice you want to clone

## Model Details

### Text-to-Speech Model
The TTS model has been fine-tuned to match TARS's voice from the Interstellar movie:

1. **Base Model**: XTTS2 (Coqui TTS)
2. **Training Data**:
   - High-quality audio clips from Interstellar movie
   - Cleaned and processed for optimal voice cloning
   - Located in `models/TTS/wavs/`
3. **Fine-tuning Process**:
   - Audio clips segmented into 6-30 second chunks
   - Background noise and music removed
   - Normalized for consistent volume levels
   - Fine-tuned using Coqui TTS training pipeline
4. **Voice Characteristics**:
   - Deep, authoritative tone
   - Clear pronunciation
   - Natural pacing and inflection
   - Emotional range for sarcasm and humor

### Language Model
The Mistral-7B model is used with specific configurations:
- Context window: 8192 tokens
- Batch size: 512 tokens
- GPU acceleration enabled
- Custom system prompt for TARS personality
- Temperature: 0.7 for balanced creativity

## Usage
1. Run the main script:
```bash
python main.py
```

2. Wait for TARS to initialize (you'll see "TARS initialized. Humor: 90%, Honesty: 90%")

3. Speak when prompted or use the wake word "Hey TARS" to start a conversation

4. Press Ctrl+C to exit the conversation

## Project Structure
```
TARS/
├── models/              # Model files
│   ├── LLM/            # Large Language Models
│   │   └── mistral-7b-instruct-v0.2.Q6_K.gguf  # LLM model
│   └── TTS/            # Text-to-Speech models
│       ├── config.json # TTS model config
│       ├── model.pth   # TTS model weights
│       └── wavs/       # Training audio files
├── .vscode/            # VS Code settings
├── audio_handler.py    # Audio recording and playback
├── llm_handler.py      # LLM interaction
├── main.py            # Main application
├── speech_handler.py   # Speech synthesis
├── wake_word_handler.py # Wake word detection
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 