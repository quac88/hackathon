import json
from client import ScaleAIClient
from utils import TokenizerWrapper
from embeddings import SentenceEmbedder
import torch
import sys  

# Configuration
API_KEY = "live_a04ee4b972384b13bac7597080aface6"  # Replace with your actual API key
MODEL_TYPES = [
    "OPENAI_GPT3.5-TURBO", "OPENAI_GPT4",
    "SCALE_FALCON_40B_INSTRUCT", "ANTHROPIC_CLAUDE",
    "SCALE_MPT_7B_CHAT", "SCALE_LLAMA_2_13B_CHAT",
    "SCALE_LLAMA_2_7B_CHAT", "SCALE_VICUNA_13B", 
    "COHERE_CHAT"
]

def run_moe_system(input_text):
    client = ScaleAIClient(API_KEY)
    tokenizer_wrapper = TokenizerWrapper("gpt2", 2048)
    sentence_embedder = SentenceEmbedder()  # Initialize your sentence embedder

    responses = []
    embeddings = []
    for model_type in MODEL_TYPES:
        response = client.forward(model_type, input_text)
        responses.append(response)

        # Generate embedding for each response
        embedding = sentence_embedder([response.get('text', '')])
        embeddings.append(embedding)

    best_response = select_best_response(responses, embeddings)
    return best_response

def select_best_response(responses, embeddings):
    # Selecting the response with the embedding closest to the average
    avg_embedding = sum(embeddings) / len(embeddings)
    best_idx = min(range(len(embeddings)), key=lambda i: torch.norm(embeddings[i] - avg_embedding))
    return responses[best_idx].get('text', '')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])  # Combine all command-line arguments into a single string
        best_answer = run_moe_system(prompt)
        print("Best Response:", best_answer)
    else:
        print("Usage: python script_name.py 'your prompt here'")
