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
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
   - Mistral-7B Model:
     - Download `mistral-7b-instruct-v0.2.Q6_K.gguf` from [TheBloke's HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
     - Place it in the `models` directory
   - TTS Model:
     - Option 1: Use the pre-trained XTTS2 model
     - Option 2: Fine-tune your own model using [AllTalk](https://github.com/erew123/alltalk_tts)
   - Reference Audio:
     - Place your reference audio file (e.g., reference.wav) in the project root
     - This should be a clear, high-quality recording of the voice you want to clone

## Usage
1. Run the main script:
```bash
python main.py
```

2. Wait for TARS to initialize (you'll see "TARS initialized. Humor: 90%, Honesty: 90%")

3. Speak when prompted or use the wake word to start a conversation

4. Press Ctrl+C to exit the conversation

## Project Structure
```
TARS/
├── models/              # Model files
│   ├── mistral-7b-instruct-v0.2.Q6_K.gguf  # LLM model
│   └── TARS/           # TTS model files
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