import os
import time
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from speech_handler import SpeechHandler
from wake_word_handler import WakeWordHandler
import sys

class TARS:
    def __init__(self, ref_audio_path=None):
        print("\nInitializing TARS...")
        self.audio_handler = AudioHandler()
        self.llm_handler = LLMHandler()
        self.speech_handler = SpeechHandler(ref_audio_path)
        self.wake_word_handler = WakeWordHandler()
        print("TARS initialized. Humor: 90%, Honesty: 90%")

    def handle_conversation_turn(self):
        """Handle a single conversation turn"""
        # Record audio
        audio_data = self.audio_handler.record_audio()
        if audio_data is None:
            return False

        # Save and transcribe audio
        temp_audio_file = self.audio_handler.save_temp_wav(audio_data)
        if temp_audio_file is None:
            return False

        transcribed_text = self.speech_handler.transcribe_audio(temp_audio_file)
        if not transcribed_text:
            print("I couldn't catch that. Could you repeat?")
            return False

        print(f"\nYou: {transcribed_text}")

        # Generate and speak response
        response, should_end = self.llm_handler.generate_response(transcribed_text)
        print(f"TARS: {response}")

        speech_file = self.speech_handler.generate_speech(response)
        if speech_file and os.path.exists(speech_file):
            audio_data = self.audio_handler.save_audio(speech_file, speech_file)
            self.audio_handler.play_audio(audio_data)
            time.sleep(0.5)

        return should_end

    def cleanup(self):
        """Cleanup resources before exit"""
        try:
            self.wake_word_handler.stop_listening()
            self.llm_handler.cleanup()  # Clean up LLM first
            print("\nShutting down TARS...")
        except Exception as e:
            print(f"\nError during cleanup: {e}")

    def conversation_loop(self):
        """Main conversation loop"""
        try:
            while True:
                if not self.wake_word_handler.start_listening():
                    continue

                print(f"\nTARS: {self.llm_handler.welcome_message}")
                
                # Speak the welcome message
                speech_file = self.speech_handler.generate_speech(self.llm_handler.welcome_message)
                if speech_file and os.path.exists(speech_file):
                    audio_data = self.audio_handler.save_audio(speech_file, speech_file)
                    self.audio_handler.play_audio(audio_data)
                    time.sleep(0.5)
                
                self.llm_handler.reset_conversation()

                while True:
                    should_end = self.handle_conversation_turn()
                    if should_end:
                        self.cleanup()
                        sys.exit(0)  # Clean exit

        except KeyboardInterrupt:
            self.cleanup()
            sys.exit(0)  # Clean exit on Ctrl+C
        except Exception as e:
            print(f"\nSystem error: {e}")
            self.cleanup()
            sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    # You can specify a reference audio file here
    ref_audio = "reference.wav"  # Replace with your reference audio file
    tars = TARS(ref_audio)
    tars.conversation_loop() 