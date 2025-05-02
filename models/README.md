# Models Directory

This directory is used to store various model files required for the TARS system. The following files should be placed in this directory:

## Required Model Files

- **TTS Models**: Text-to-Speech model files (`.pth`, `.pt`, or `.ckpt` files)
- **LLM Models**: Large Language Model files (`.gguf` files)
- **Voice Models**: Voice model files (`.bin`, `.onnx`, or `.safetensors` files)

## Directory Structure

```
models/
├── LLM/              # Large Language Models
│   └── mistral-7b-instruct-v0.2.Q6_K.gguf  # Mistral model for text generation
├── TTS/              # Text-to-Speech models
│   ├── config.json   # TTS model configuration (included in repo)
│   ├── model.pth     # TTS model weights (needs to be downloaded)
│   └── wavs/         # Training audio files (included in repo)
└── Voice/            # Voice models
```

## Model Details

### LLM Models
- **Mistral-7B**: The default language model used by TARS
  - File: `mistral-7b-instruct-v0.2.Q6_K.gguf`
  - Source: [TheBloke's HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
  - Place in: `models/LLM/`
  - Status: Not included in repository (needs to be downloaded)

### TTS Models
- **XTTS2**: Text-to-Speech model for voice synthesis
  - Files:
    - `config.json`: Included in repository
    - `model.pth`: Needs to be downloaded
  - Place in: `models/TTS/`
  - Training audio files in `models/TTS/wavs/` are included in repository

## Notes

- Most model files are gitignored to prevent large files from being tracked
- The `models/TARS` directory and its configuration files are included in the repository
- Keep your model files organized in their respective subdirectories
- Make sure to download and place the required model files before running the system
- Model files are not included in the repository due to their size and licensing restrictions

## Model Sources

Please obtain the required model files from their official sources or as specified in the project documentation. 