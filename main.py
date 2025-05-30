
from flask import Flask, request, jsonify
import numpy as np
from utils import preprocess_fft, preprocess_fft_xyz

from sklearn.externals import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
logistic_model = joblib.load('model/logistic_model.pkl')
forest_model = joblib.load('model/forest_model.pkl')
lstm_model = load_model('model/lstm_model.h5')

@app.route('/')
def home():
    return "MagPulse API is running with multiple models."

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    data = request.get_json()
    fft_input = np.array(data['fft_data']).reshape(1, -1)
    prediction = int(logistic_model.predict(fft_input)[0])
    prob = float(logistic_model.predict_proba(fft_input)[0][1])
    return jsonify({'model': 'logistic', 'prediction': prediction, 'confidence': prob})

@app.route('/predict/forest', methods=['POST'])
def predict_forest():
    data = request.get_json()
    fft_input = np.array(data['fft_data']).reshape(1, -1)
    prediction = int(forest_model.predict(fft_input)[0])
    prob = float(forest_model.predict_proba(fft_input)[0][1])
    return jsonify({'model': 'random_forest', 'prediction': prediction, 'confidence': prob})

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    data = request.get_json()
    fft_input = np.array(data['fft_data'])
    X = preprocess_fft(fft_input)
    y_pred = lstm_model.predict(X)[0]
    prediction = int(np.argmax(y_pred))
    confidence = float(np.max(y_pred))
    return jsonify({'model': 'lstm', 'prediction': prediction, 'confidence': confidence})

@app.route('/predict/cnn', methods=['POST'])
def predict_cnn():
    return jsonify({'model': 'cnn', 'status': 'not implemented yet'})


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    data = request.get_json()
    fft_data = data.get("fft_data", [])
    true_label = data.get("true_label")
    model_used = data.get("model")

    if not fft_data or true_label is None:
        return jsonify({"error": "Missing data"}), 400

    # Append feedback to CSV for later retraining
    with open("feedback_log.csv", "a") as f:
        row = ",".join(map(str, fft_data)) + f",{true_label},{model_used}\n"
        f.write(row)

    return jsonify({"status": "feedback saved"})

@app.route('/approve', methods=['POST'])
def approve_prediction():
    data = request.get_json()
    fft_data = data.get("fft_data", [])
    approved = data.get("approved", False)
    model_used = data.get("model")
    prediction = data.get("prediction")

    if not fft_data or prediction is None or model_used is None:
        return jsonify({"error": "Missing data"}), 400

    # Log approval/rejection for audit and adaptive training
    with open("approval_log.csv", "a") as f:
        row = ",".join(map(str, fft_data)) + f",{prediction},{approved},{model_used}\n"
        f.write(row)

    return jsonify({"status": "approval logged"})


@app.route('/retry_predict', methods=['POST'])
def retry_predict():
    data = request.get_json()
    fft_data_batches = data.get("fft_batches")  # list of peak candidates
    model_names = ["logistic", "forest", "lstm"]
    responses = []

    if not fft_data_batches or not isinstance(fft_data_batches, list):
        return jsonify({"error": "fft_batches must be a list"}), 400

    for i, fft_data in enumerate(fft_data_batches):
        batch_result = {"peak_index": i, "results": []}
        fft_array = np.array(fft_data).reshape(1, -1)

        # Logistic
        try:
            log_pred = int(logistic_model.predict(fft_array)[0])
            log_conf = float(logistic_model.predict_proba(fft_array)[0][1])
            batch_result["results"].append({"model": "logistic", "prediction": log_pred, "confidence": log_conf})
        except Exception as e:
            batch_result["results"].append({"model": "logistic", "error": str(e)})

        # Forest
        try:
            for_pred = int(forest_model.predict(fft_array)[0])
            for_conf = float(forest_model.predict_proba(fft_array)[0][1])
            batch_result["results"].append({"model": "forest", "prediction": for_pred, "confidence": for_conf})
        except Exception as e:
            batch_result["results"].append({"model": "forest", "error": str(e)})

        # LSTM
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
