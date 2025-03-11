from llama_cpp import Llama

MODEL_PATH = "mistral/mistral-7b-instruct-v0.2.Q6_K.gguf"

# Load model with GPU acceleration
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # Assign 50 layers to the GPU
    n_threads=8,  # Use 8 CPU threads for token decoding
    n_batch=512,  # Optimize batch size for faster inference
)

# Run a test inference
response = llm("Hello, how are you?", max_tokens=50)
print(response["choices"][0]["text"])
