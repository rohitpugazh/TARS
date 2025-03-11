# **TARS-Inspired Conversational AI**

## **Project Overview**  
This project is a **TARS-inspired Conversational AI** that replicates the voice and conversational style of TARS from *Interstellar*. The AI assistant processes **spoken input, generates intelligent responses, and converts text back to speech**, using a **fine-tuned XTTSv2 model** to mimic *Bill Irwin’s voice*.

---

## **Pipeline Overview**  
1. **Speech Recognition (ASR)** – Uses **OpenAI Whisper** to transcribe user speech.  
2. **Conversational AI** – Implements a **local LLaMA-based Mistral-7B model** for multi-turn, context-aware responses.  
3. **Text-to-Speech (TTS)** – Utilizes **XTTSv2**, fine-tuned to sound like *TARS*, for realistic speech synthesis.  

---

## **Features**  
- **Real-time Speech Recognition** with **OpenAI Whisper**.  
- **Locally hosted Mistral-7B (LLaMA-based) model** for **context-aware dialogue generation**.  
- **Custom voice synthesis** trained on Bill Irwin’s voice using **XTTSv2**.  
- **Low-latency AI processing pipeline** optimized with **Torch & FFmpeg**.  
- **Fully autonomous AI assistant**, capable of multi-turn conversations with **stored context**.  

---

## **Tech Stack & Tools**  
- **Python 3.12.9**  
- **Torch, Transformers, Whisper, llama.cpp, TTS (XTTSv2)**  
- **FFmpeg (for audio processing)**  
- **GPU-accelerated LLaMA-based model (Mistral-7B)**  

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/TARS-Voice-Assistant.git
cd TARS-Voice-Assistant
```

### **2. Install Dependencies**  
Install all dependencies from the `requirements.txt` file:  
```bash
pip install -r requirements.txt
```

### **3. Configure System Paths**  
Ensure **FFmpeg** is installed and added to the system path. On Windows, run:  
```bash
setx PATH "%PATH%;C:\\ffmpeg\\bin"
```

### **4. Run the Assistant**  
```bash
python main.py
```

---

## **Usage**  
1. **Provide an audio file** (`input.wav`) containing speech input.  
2. The system will **transcribe** the speech using **Whisper ASR**.  
3. The transcribed text is **fed into the Mistral-7B model**, which generates a response.  
4. The AI-generated text is **converted back to speech** using **XTTSv2**.  
5. The output audio file (`output_speech.wav`) is saved and played.  

---

## **File Structure**  
```
TARS-Voice-Assistant/
|
|-- LICENSE                 # License file
|-- README.md               # Project documentation
|-- requirements.txt        # Required dependencies
|
|-- main.py                 # Entry point for running the assistant
|-- clean_voice.py          # Audio preprocessing and noise reduction
|-- preprocess_wavs.py      # Audio preprocessing pipeline
|-- trim_audio.py           # Trims audio files to desired length
|
|-- mistral/                # Mistral-7B conversational AI model
|-- tts_model/              # XTTSv2-based text-to-speech model
|
|-- input.wav               # Example input speech file
|-- output_speech.wav       # Output speech file after TTS processing
|
|-- mistral_test.py         # Test script for Mistral AI model
|-- test_voice_cloning.py   # Test script for voice cloning
|-- scraper.py              # Web scraping utility (if needed for datasets)
```

---

## **Customization**  
- **Change Whisper Model**: Modify `model_size="turbo"` in `SpeechRecognizer`.  
- **Adjust AI Response Behavior**: Modify the `system` role in `ConversationalAI`.  
- **Use a Different Voice**: Replace `reference.wav` with a new **voice reference** for XTTSv2.  

---

## **Resources & References**  
- [OpenAI Whisper](https://github.com/openai/whisper) - Automatic Speech Recognition (ASR)  
- [Mistral-7B (LLaMA-based)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) - Local Conversational AI Model  
- [XTTSv2](https://github.com/coqui-ai/TTS) - Advanced Text-to-Speech Model  
- [AllTalk Fine-Tuning Guide](https://github.com/coqui-ai/TTS) - Instructions for fine-tuning XTTSv2  
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) - Running LLaMA models efficiently on local hardware  

---

## **Future Enhancements**  
- Live microphone input & real-time processing.  
- Web-based or mobile interface integration.  
- Fine-tuned conversational memory for long-term context retention.  

---

## **License**  
MIT License - Feel free to use and modify!  

---

