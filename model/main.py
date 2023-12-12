import json
from client import ScaleAIClient
from utils import TokenizerWrapper  # If you're using this for preprocessing
from requests.auth import HTTPBasicAuth

# Configuration
API_KEY = "live_a04ee4b972384b13bac7597080aface6"  # Replace with your actual API key
MODEL_TYPES: list[str] = ["OPENAI_GPT3.5-TURBO", "OPENAI_GPT4"]  # Models to use

def run_moe_system(input_text):
    # Initialize the Scale AI client
    client = ScaleAIClient(API_KEY)

    # Initialize the tokenizer with the appropriate model name and sequence length
    tokenizer_wrapper = TokenizerWrapper("gpt2", 512)  # Replace "gpt2" with your model name

    # Tokenize the input text
    preprocessed_input = tokenizer_wrapper(input_text)

    # Collect responses from each model
    responses = []
    for model_type in MODEL_TYPES:
        # Send the raw input text to the API
        response = client.forward(model_type, input_text)
        print(f"Response from {model_type}: {response}")  # Debug print
        responses.append(response)

    # Aggregate the responses
    aggregated_response = aggregate_responses(responses)

    return aggregated_response


def aggregate_responses(responses):
    # Implement your logic to aggregate responses from different models
    # For example, you might concatenate responses, choose the best one, etc.
    # This is a simple example of concatenating responses
    return " ".join([resp['response'] for resp in responses])

if __name__ == "__main__":
    input_text = "Your input text here"  # Replace with the text you want to process
    result = run_moe_system(input_text)
    print("Aggregated Response:", result)
