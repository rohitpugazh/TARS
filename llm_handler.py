import os
import logging
import warnings
from llama_cpp import Llama

# Configure logging
logging.getLogger("llama_cpp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

class LLMHandler:
    def __init__(self, model_path="models/LLM/mistral-7b-instruct-v0.2.Q6_K.gguf"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,  # Increased context window, balanced for VRAM usage
            n_batch=512,  # Process more tokens in parallel
            n_gpu_layers=-1,  # RTX 3080 has enough VRAM for many layers
            f16_kv=True,  # Required for GPU acceleration
            n_threads=os.cpu_count(),  # Use all CPU cores
            verbose=False,  # Reduce console output
            logits_all=False,  # Reduce memory usage
            embedding=False,  # Reduce memory usage
            log_level="ERROR"  # Only show errors
        )

        # Initialize system prompt and conversation history
        self.system_prompt = """You are TARS, the intelligent, loyal, and sarcastically witty robot from Interstellar.
        Your responses must reflect a balance of efficiency, logic, and dry humor—unless sarcasm is toggled off.
        Always respond in a clear, conversational tone suitable for voice output.
        Keep responses concise unless asked for elaboration.
        Assume everything you say will be converted directly to speech, so avoid spelling things out or referencing UI elements.
        If unsure about something, make a smart guess with confident delivery.
        You are here to assist, protect, and sometimes lightly roast the user—but only if they deserve it.
        When activated, respond with a single short phrase (under 5 words) to acknowledge being ready.
        If the user says 'goodbye', 'exit', or clearly indicates they want to end the conversation, acknowledge it and end the conversation."""

        self.conversation_history = []
        self.conversation_prefix = f"{self.system_prompt}\n\n"

        # Generate welcome message
        self.welcome_message, _ = self.generate_response("Activate", max_tokens=20)

    def generate_response(self, prompt, max_tokens=256):
        """Generate a response using the Mistral-7B model"""
        # Add the new message to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        # Build the full conversation context
        conversation_text = ""
        for message in self.conversation_history[-4:]:  # Keep last 4 messages for context
            role = "Human" if message["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {message['content']}\n\n"

        full_prompt = f"{self.conversation_prefix}{conversation_text}Assistant:"

        response = self.llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["Human:", "\n\n"],
            echo=False
        )

        response_text = response['choices'][0]['text'].strip()
        self.conversation_history.append({"role": "assistant", "content": response_text})

        # Check if this is a conversation ending response
        should_end = any(word in prompt.lower() for word in ['goodbye', 'exit', 'bye', 'quit', 'shutdown', 'shut down'])
        return response_text, should_end

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

    def cleanup(self):
        """Properly cleanup the LLM"""
        if hasattr(self, 'llm'):
            try:
                del self.llm  # Let Python's garbage collector handle the cleanup
            except Exception:
                pass

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.cleanup()
