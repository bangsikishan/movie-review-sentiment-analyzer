import os

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

CORS(app)

# Load model and tokenizer once when the app starts
MODEL_PATH = ""  # Path to your sentiment analysis model folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


def predict_sentiment(review_text, model=model, tokenizer=tokenizer, device=device):
    """
    Predict sentiment for a given review text.

    Args:
        review_text (str): The text review to analyze
        model (transformers.AutoModelForSequenceClassification): The loaded model
        tokenizer (transformers.AutoTokenizer): The tokenizer used to process the text
        device (torch.device): The device (CPU or CUDA) to run the model

    Returns:
        dict: Dictionary containing the prediction ('positive' or 'negative') and confidence score
    """
    # Tokenize input
    inputs = tokenizer(
        review_text, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction[0]].item()

    # Convert prediction to label
    sentiment = "positive" if prediction.item() == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "negative": probabilities[0][0].item(),
            "positive": probabilities[0][1].item(),
        },
    }


@app.route("/")
def index():
    return send_from_directory(os.getcwd(), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review_text = data.get("review_text", "")
        print(review_text)
        if not review_text:
            return jsonify({"error": "Review text is required"}), 400

        # Get sentiment prediction
        prediction = predict_sentiment(review_text)

        return jsonify(prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
