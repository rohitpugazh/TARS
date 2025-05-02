# Models Directory

This directory is used to store various model files required for the TARS system. The following files should be placed in this directory:

## Required Model Files

- **TTS Models**: Text-to-Speech model files (`.pth`, `.pt`, or `.ckpt` files)
- **LLM Models**: Large Language Model files (`.gguf` files)

## Directory Structure

```
models/
├── LLM/              # Large Language Models
│   └── mistral-7b-instruct-v0.2.Q6_K.gguf  # Mistral model for text generation
├── TTS/              # Text-to-Speech models
│   ├── config.json   # TTS model configuration
│   ├── model.pth     # TTS model weights
│   └── wavs/         # Training audio files
```

## Model Details

### LLM Models
- **Mistral-7B**: The default language model used by TARS
  - File: `mistral-7b-instruct-v0.2.Q6_K.gguf`
  - Source: [TheBloke's HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
  - Place in: `models/LLM/`

### TTS Models
- **XTTS2**: Text-to-Speech model for voice synthesis
  - Files: `config.json` and `model.pth`
  - Place in: `models/TTS/`
  - Training audio files go in: `models/TTS/wavs/`

## Notes

- This directory is gitignored to prevent large model files from being tracked
- Keep your model files organized in their respective subdirectories
- Make sure to download and place the required model files before running the system
- Model files are not included in the repository due to their size and licensing restrictions

## Model Sources

Please obtain the required model files from their official sources or as specified in the project documentation. 