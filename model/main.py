import json
from client import ScaleAIClient
from utils import TokenizerWrapper

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

    responses = []
    for model_type in MODEL_TYPES:
        response = client.forward(model_type, input_text)
        responses.append(response)

    best_response = select_best_response(responses)
    return best_response

def select_best_response(responses):
    # Implement your logic to select the best response
    # This is a placeholder for your selection criteria
    # Example: Selecting the longest response
    best = max(responses, key=lambda resp: len(resp.get('text', '')))
    return best.get('text', '')

if __name__ == "__main__":
    input_text = "Tell me fun facts about ants"  # Replace with the text you want to process
    best_answer = run_moe_system(input_text)
    print("Best Response:", best_answer)
