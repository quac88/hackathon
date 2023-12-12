from flask import Flask, request, jsonify
from main import run_moe_system

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    input_data = request.get_json()
    prompt = input_data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    best_answer = run_moe_system(prompt)
    
    return jsonify({"response": best_answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
