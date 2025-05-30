
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

if __name__ == '__main__':
    app.run(debug=True)
