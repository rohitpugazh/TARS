"""
Conversational AI Pipeline: 
1. Speech Recognition (Whisper ASR) 
2. Conversational AI (ChatGPT) 
3. Text-to-Speech (XTTSv2)
"""

import os
import time
import logging
import torch
import torchaudio
import whisper
from llama_cpp import Llama
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"  # Add FFmpeg to system path
INPUT_AUDIO = "input.wav"  # Audio file to be transcribed
OUTPUT_AUDIO = "output_speech.wav"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AI_MODEL = "mistral/mistral-7b-instruct-v0.2.Q6_K.gguf"
CONFIG_PATH = "tts_model/config.json"  # Config file for fine-tuned model
TOKENIZER_PATH = "tts_model/vocab.json"  # Tokenizer file used during training
XTTS_CHECKPOINT = "tts_model/model.pth"  # Fine-tuned model checkpoint
REFERENCE_VOICE = "reference.wav"  # Reference audio file for speaker adaptation


class SpeechRecognizer:
    """Speech recognition using OpenAI Whisper."""

    def __init__(self, model_size: str = "turbo") -> None:
        """
        Initialize the speech recognizer.

        :param model_size: Size of the Whisper model to use 
        (e.g., 'tiny', 'base', 'medium', 'large').
        """
        logging.info("Initializing Whisper model (%s) on device: %s", model_size, DEVICE)
        self.model = whisper.load_model(model_size).to(DEVICE)

    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe the given audio file.

        :param audio_file: Path to the audio file.
        :return: Transcribed text.
        :raises FileNotFoundError: If the audio file does not exist.
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        logging.info("Transcribing audio...")
        result = self.model.transcribe(audio_file)
        return result.get("text", "")


class ConversationalAI:
    """Conversational AI using a locally running Mistral-7B model with GPU acceleration."""

    def __init__(self) -> None:
        """Initialize the local LLaMA model."""
        logging.info("Loading local chat model with GPU acceleration...")

        # Load model with GPU acceleration
        self.llm = Llama(
            model_path=AI_MODEL,
            n_ctx=4096,  # Context length (can go higher)
            n_gpu_layers=50,  # Load 50 layers into GPU for fast inference
            n_batch=512,  # Increase batch size for efficiency
            n_threads=8,  # Number of threads for parallel processing
            n_gpu=1,  # Number of GPUs to use
        )

        # Store chat history for multi-turn conversations
        self.conversation_history = [
            {"role": "system",
             "content": "You are an AI assistant who talks like TARS from Interstellar. Make jokes occasionally."}
        ]

    def generate_response(self, user_input: str) -> str:
        """
        Generate a conversational response using the local model.

        :param user_input: User's transcribed speech.
        :return: AI-generated response.
        """
        logging.info("Generating AI response...")

        # Append user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Format history as a chat prompt
        formatted_prompt = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history]
        )

        # Run inference
        output = self.llm(
            formatted_prompt,
            max_tokens=512,  # Increase for longer responses
            stop=["User:", "System:"],  # Prevent hallucination
            echo=False,
        )

        response_text = output["choices"][0]["text"].strip()
        # Append AI response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text


class TextToSpeech:
    """Text-to-Speech (TTS) using the fine-tuned XTTSv2 model with reference voice."""

    def __init__(self) -> None:
        """
        Initialize the fine-tuned XTTSv2 model.
        """
        logging.info("Loading fine-tuned XTTSv2 model from local directory...")
        self.config = XttsConfig()
        self.config.load_json(CONFIG_PATH)
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
        self.model.to(DEVICE)

    def generate_speech(self, input_text: str, output_wav: str) -> None:
        """
        Convert text to speech using the fine-tuned XTTSv2 model with a reference voice.
        
        :param input_text: Text to synthesize into speech.
        :param output_wav: Path to save the output speech WAV file.
        :raises FileNotFoundError: If the reference voice file is missing.
        """
        if not os.path.exists(REFERENCE_VOICE):
            raise FileNotFoundError(f"Reference voice file not found: {REFERENCE_VOICE}")

        logging.info("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[REFERENCE_VOICE])

        logging.info("Synthesizing speech with fine-tuned XTTSv2...")
        start_time = time.time()

        # Perform inference
        result = self.model.inference(
            input_text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7,  # Add custom parameters here
        )

        duration = time.time() - start_time
        logging.info("Speech synthesis completed in %.2f seconds", duration)

        # Save output audio
        torchaudio.save(output_wav, torch.tensor(result["wav"]).unsqueeze(0), 24000)
        logging.info("Speech saved to: %s", output_wav)


if __name__ == "__main__":
    try:
        recognizer = SpeechRecognizer(model_size="turbo")
        chatbot = ConversationalAI()
        tts = TextToSpeech()

        # Step 1: Speech-to-Text (ASR)
        transcribed_text = recognizer.transcribe(INPUT_AUDIO)
        logging.info("Transcribed Text: %s", transcribed_text)

        # Step 2: Conversational AI Response
        ai_response = chatbot.generate_response(transcribed_text)
        logging.info("TARS: %s", ai_response)

        # Step 3: Text-to-Speech (TTS)
        tts.generate_speech(ai_response, OUTPUT_AUDIO)

    except FileNotFoundError as e:
        logging.error(e)
