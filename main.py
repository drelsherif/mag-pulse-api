

from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils import preprocess_fft

app = Flask(__name__)

# Load models
logistic_model = joblib.load("model/logistic_model.pkl")
forest_model = joblib.load("model/forest_model.pkl")
lstm_model = load_model("model/lstm_model.h5")

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    data = request.get_json()
    fft_data = np.array(data["fft_data"]).reshape(1, -1)
    prediction = int(logistic_model.predict(fft_data)[0])
    confidence = float(logistic_model.predict_proba(fft_data)[0][1])
    return jsonify({"model": "logistic", "prediction": prediction, "confidence": confidence})

@app.route('/predict/forest', methods=['POST'])
def predict_forest():
    data = request.get_json()
    fft_data = np.array(data["fft_data"]).reshape(1, -1)
    prediction = int(forest_model.predict(fft_data)[0])
    confidence = float(forest_model.predict_proba(fft_data)[0][1])
    return jsonify({"model": "forest", "prediction": prediction, "confidence": confidence})

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    data = request.get_json()
    fft_data = preprocess_fft(data["fft_data"])
    output = lstm_model.predict(fft_data)[0]
    prediction = int(np.argmax(output))
    confidence = float(np.max(output))
    return jsonify({"model": "lstm", "prediction": prediction, "confidence": confidence})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    # Save to log file or database for retraining later
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"status": "feedback recorded"})

@app.route('/approve', methods=['POST'])
def approve():
    data = request.get_json()
    # Save approved predictions for learning
    with open("approval_log.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"status": "approval recorded"})

@app.route('/retry_predict', methods=['POST'])
def retry_predict():
    data = request.get_json()
    fft_data_batches = data.get("fft_batches")
    model_names = ["logistic", "forest", "lstm"]
    responses = []

    if not fft_data_batches or not isinstance(fft_data_batches, list):
        return jsonify({"error": "fft_batches must be a list"}), 400

    for i, fft_data in enumerate(fft_data_batches):
        batch_result = {"peak_index": i, "results": []}
        fft_array = np.array(fft_data).reshape(1, -1)

        try:
            log_pred = int(logistic_model.predict(fft_array)[0])
            log_conf = float(logistic_model.predict_proba(fft_array)[0][1])
            batch_result["results"].append({"model": "logistic", "prediction": log_pred, "confidence": log_conf})
        except Exception as e:
            batch_result["results"].append({"model": "logistic", "error": str(e)})

        try:
            for_pred = int(forest_model.predict(fft_array)[0])
            for_conf = float(forest_model.predict_proba(fft_array)[0][1])
            batch_result["results"].append({"model": "forest", "prediction": for_pred, "confidence": for_conf})
        except Exception as e:
            batch_result["results"].append({"model": "forest", "error": str(e)})

        try:
            lstm_input = preprocess_fft(fft_data)
            lstm_out = lstm_model.predict(lstm_input)[0]
            lstm_pred = int(np.argmax(lstm_out))
            lstm_conf = float(np.max(lstm_out))
            batch_result["results"].append({"model": "lstm", "prediction": lstm_pred, "confidence": lstm_conf})
        except Exception as e:
            batch_result["results"].append({"model": "lstm", "error": str(e)})

        responses.append(batch_result)

    return jsonify({"retries": responses})

if __name__ == '__main__':
    app.run(debug=True)
