import json
import requests
from requests.auth import HTTPBasicAuth

class ScaleAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.donovan.scale.com/v1/chat"
        self.auth = HTTPBasicAuth(self.api_key, '')  # Using HTTP Basic Authentication

    def forward(self, model_type, input_text):
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        data = {
            "modelType": model_type,
            "text": input_text,
            "workspace": "atrjw4p7vy9zi9p7kov0a40v",
            "thread": "thread"
        }
        response = requests.post(self.api_url, headers=headers, json=data, auth=self.auth)
        return response.json()


class MakotoClient:
    def __init__(self, api_key):
        self.scale_ai_client = ScaleAIClient(api_key)

    def forward(self, input_text):
        # Using Scale AI client to send the request and get the response
        response = self.scale_ai_client.forward(input_text)
        # Process the response as needed, e.g., extract text, logits, etc.
        # This will depend on the response format from Scale AI
        return response

if __name__ == "__main__":
    api_key = "live_a04ee4b972384b13bac7597080aface6"  # Replace with your actual API key
    client = MakotoClient(api_key)
    input_text = "Your input text here"
    response = client.forward(input_text)
    print(response)
