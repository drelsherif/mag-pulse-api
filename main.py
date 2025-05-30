
from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from utils import preprocess_fft, preprocess_fft_xyz

app = Flask(__name__)

# Load models
logistic_model = joblib.load("model/logistic_model.pkl")
forest_model = joblib.load("model/forest_model.pkl")
lstm_model = tf.keras.models.load_model("model/lstm_model.h5")

# In-memory model queue
model_queue = [("logistic", logistic_model), ("forest", forest_model), ("lstm", lstm_model)]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    fft = data["fft"]
    model_name = data.get("model", "logistic")

    if model_name == "logistic":
        X = preprocess_fft(fft)
        pred = logistic_model.predict(X.reshape(1, -1))[0]
    elif model_name == "forest":
        X = preprocess_fft(fft)
        pred = forest_model.predict(X.reshape(1, -1))[0]
    elif model_name == "lstm":
        X = preprocess_fft(fft)
        pred = np.argmax(lstm_model.predict(X), axis=1)[0]
    else:
        return jsonify({"error": "Invalid model specified"}), 400

    return jsonify({"model": model_name, "prediction": int(pred)})

@app.route("/retry_predict", methods=["POST"])
def retry_predict():
    data = request.json
    fft = data["fft"]
    rejected_models = data.get("rejected", [])

    for name, model in model_queue:
        if name in rejected_models:
            continue

        if name in ["logistic", "forest"]:
            X = preprocess_fft(fft)
            pred = model.predict(X.reshape(1, -1))[0]
        elif name == "lstm":
            X = preprocess_fft(fft)
            pred = np.argmax(model.predict(X), axis=1)[0]
        else:
            continue

        return jsonify({
            "model": name,
            "prediction": int(pred),
            "rejected": rejected_models
        })

    return jsonify({"error": "No remaining models to try"}), 400

@app.route("/")
def index():
    return "MagPulse API is running"

if __name__ == "__main__":
    app.run(debug=True)
