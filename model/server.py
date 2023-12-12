import requests
from flask import Flask, request, jsonify
from typing import NamedTuple, Dict

app = Flask(__name__)

# API configuration
SCALE_AI_ENDPOINT = "https://api.donovan.scale.com/v1/chat"
SCALE_AI_API_KEY = "live_a04ee4b972384b13bac7597080aface6"  # Replace with your actual API key

class MakotoResponse(NamedTuple):
    response_text: str


def query_model_api(model_type, text):
    payload = {
        "modelType": model_type,
        "text": text,
        "workspace": "atrjw4p7vy9zi9p7kov0a40v",
        "thread": "thread"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {SCALE_AI_API_KEY}"
    }
    response = requests.post(SCALE_AI_ENDPOINT, json=payload, headers=headers)
    return response.json()


@app.route("/forward", methods=["POST"])
def forward():
    input_data = request.json
    text = input_data["text"]

    # Querying both models
    response_gpt35 = query_model_api('OPENAI_GPT3.5-TURBO', text)
    response_gpt4 = query_model_api('OPENAI_GPT4', text)

    # Aggregation logic (this is a placeholder, modify as needed)
    # For simplicity, we concatenate the responses from both models
    aggregated_response = response_gpt35['response'] + " " + response_gpt4['response']

    return jsonify(MakotoResponse(response_text=aggregated_response))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
