
from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from utils import preprocess_fft, preprocess_fft_xyz
import time
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models and training data storage
logistic_model = None
forest_model = None
lstm_model = None
training_data = []  # In-memory storage for training data

def load_models():
    """Load all ML models with error handling"""
    global logistic_model, forest_model, lstm_model
    
    try:
        logistic_model = joblib.load("model/logistic_model.pkl")
        logger.info("Logistic model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load logistic model: {e}")
        
    try:
        forest_model = joblib.load("model/forest_model.pkl")
        logger.info("Forest model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load forest model: {e}")
        
    try:
        lstm_model = tf.keras.models.load_model("model/lstm_model.h5")
        logger.info("LSTM model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {e}")

# Load models on startup
load_models()

# Model queue for retry functionality
def get_model_queue():
    """Get available models for queue processing"""
    queue = []
    if logistic_model is not None:
        queue.append(("logistic", logistic_model))
    if forest_model is not None:
        queue.append(("forest", forest_model))
    if lstm_model is not None:
        queue.append(("lstm", lstm_model))
    return queue

def validate_prediction_request(data):
    """Validate incoming prediction request"""
    if not data:
        return False, "No JSON data provided"
    
    if "fft" not in data:
        return False, "Missing 'fft' field in request"
    
    if not isinstance(data["fft"], list) or len(data["fft"]) == 0:
        return False, "FFT data must be a non-empty list"
    
    # Validate FFT data contains numbers
    try:
        fft_array = np.array(data["fft"], dtype=float)
        if np.any(np.isnan(fft_array)) or np.any(np.isinf(fft_array)):
            return False, "FFT data contains invalid values (NaN or Inf)"
    except (ValueError, TypeError):
        return False, "FFT data must contain valid numbers"
    
    return True, "Valid"

def make_prediction(fft_data, model_name):
    """Make prediction with proper error handling and timing"""
    start_time = time.time()
    
    try:
        if model_name == "logistic" and logistic_model is not None:
            X = preprocess_fft(fft_data)
            # Flatten for sklearn models and pad/truncate to expected size
            X_flat = X.flatten()
            X_padded = pad_or_truncate_features(X_flat, target_size=100)
            X_input = X_padded.reshape(1, -1)
            
            pred_proba = logistic_model.predict_proba(X_input)[0]
            pred = logistic_model.predict(X_input)[0]
            confidence = float(np.max(pred_proba))
            
        elif model_name == "forest" and forest_model is not None:
            X = preprocess_fft(fft_data)
            # Flatten for sklearn models and pad/truncate to expected size
            X_flat = X.flatten()
            X_padded = pad_or_truncate_features(X_flat, target_size=100)
            X_input = X_padded.reshape(1, -1)
            
            pred_proba = forest_model.predict_proba(X_input)[0]
            pred = forest_model.predict(X_input)[0]
            confidence = float(np.max(pred_proba))
            
        elif model_name == "lstm" and lstm_model is not None:
            # Use simpler processing for LSTM to avoid memory issues
            X = np.array(fft_data, dtype=np.float32)
            
            # Normalize
            if np.max(np.abs(X)) > 0:
                X = X / np.max(np.abs(X))
            
            # Pad or truncate to consistent size for LSTM
            X_padded = pad_or_truncate_features(X, target_size=64)
            
            # Reshape for LSTM: (batch_size, timesteps, features)
            X_input = X_padded.reshape(1, 64, 1)
            
            # Predict with reduced memory usage
            pred_proba = lstm_model.predict(X_input, verbose=0, batch_size=1)[0]
            pred = np.argmax(pred_proba)
            confidence = float(np.max(pred_proba))
            
        else:
            return None, f"Model '{model_name}' not available or not loaded"
        
        processing_time = time.time() - start_time
        
        # Convert prediction to string format expected by iOS
        prediction_str = "approved" if pred == 1 else "rejected"
        
        return {
            "prediction": prediction_str,
            "confidence": confidence,
            "model_used": model_name,
            "processing_time": processing_time,
            "features_extracted": len(fft_data)
        }, None
        
    except Exception as e:
        logger.error(f"Prediction error with {model_name}: {str(e)}")
        return None, f"Prediction failed: {str(e)}"

def pad_or_truncate_features(features, target_size):
    """Pad with zeros or truncate features to match expected model input size"""
    try:
        features = np.array(features, dtype=np.float32)
        
        if len(features) == target_size:
            return features
        elif len(features) < target_size:
            # Pad with zeros
            padding = np.zeros(target_size - len(features), dtype=np.float32)
            return np.concatenate([features, padding])
        else:
            # Truncate to target size
            return features[:target_size]
            
    except Exception as e:
        logger.error(f"Error in pad_or_truncate_features: {e}")
        return np.zeros(target_size, dtype=np.float32)

@app.route("/", methods=["GET"])
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "MagPulse API is running",
        "models_loaded": {
            "logistic": logistic_model is not None,
            "forest": forest_model is not None,
            "lstm": lstm_model is not None
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        # Validate request
        is_valid, error_msg = validate_prediction_request(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        fft_data = data["fft"]
        model_name = data.get("model", "logistic")
        session_id = data.get("session_id", "unknown")
        
        logger.info(f"Prediction request - Model: {model_name}, Session: {session_id}, FFT size: {len(fft_data)}")
        
        # Validate model name
        available_models = ["logistic", "forest", "lstm"]
        if model_name not in available_models:
            return jsonify({
                "error": f"Invalid model specified. Available models: {available_models}"
            }), 400
        
        # Make prediction
        result, error = make_prediction(fft_data, model_name)
        
        if error:
            return jsonify({"error": error}), 500
        
        logger.info(f"Prediction successful - {model_name}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/retry_predict", methods=["POST"])
def retry_predict():
    """Retry prediction with different models"""
    try:
        data = request.get_json()
        
        # Validate request
        is_valid, error_msg = validate_prediction_request(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        fft_data = data["fft"]
        rejected_models = data.get("rejected", [])
        session_id = data.get("session_id", "unknown")
        
        logger.info(f"Retry prediction - Session: {session_id}, Rejected: {rejected_models}")
        
        # Try models in order, skipping rejected ones
        model_queue = get_model_queue()
        
        for model_name, model_obj in model_queue:
            if model_name in rejected_models:
                continue
            
            result, error = make_prediction(fft_data, model_name)
            
            if not error:
                result["rejected_models"] = rejected_models
                logger.info(f"Retry successful with {model_name}: {result['prediction']}")
                return jsonify(result)
        
        return jsonify({
            "error": "No remaining models to try",
            "rejected_models": rejected_models
        }), 400
        
    except Exception as e:
        logger.error(f"Retry prediction error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/train", methods=["POST"])
def train():
    """Submit training data endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["fft_data", "label", "session_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Extract training data
        fft_data = data["fft_data"]
        label = data["label"]
        session_id = data["session_id"]
        confidence_score = data.get("confidence_score")
        features = data.get("features")
        metadata = data.get("metadata", {})
        
        # Validate FFT data
        if not isinstance(fft_data, list) or len(fft_data) == 0:
            return jsonify({"error": "fft_data must be a non-empty list"}), 400
        
        # Validate label
        valid_labels = ["approved", "rejected", "needs_review", "pending"]
        if label not in valid_labels:
            return jsonify({
                "error": f"Invalid label. Must be one of: {valid_labels}"
            }), 400
        
        # Store training data
        training_sample = {
            "fft_data": fft_data,
            "label": label,
            "session_id": session_id,
            "confidence_score": confidence_score,
            "features": features,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "sample_id": f"{session_id}_{len(training_data)}"
        }
        
        training_data.append(training_sample)
        
        logger.info(f"Training data added - Session: {session_id}, Label: {label}, Total samples: {len(training_data)}")
        
        # TODO: Here you would trigger actual model retraining
        # For now, we just store the data
        
        return jsonify({
            "message": "Training data submitted successfully",
            "samples_added": 1,
            "total_samples": len(training_data),
            "model_updated": None  # Would be set after actual retraining
        })
        
    except Exception as e:
        logger.error(f"Training endpoint error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/train_batch", methods=["POST"])
def train_batch():
    """Submit multiple training samples at once"""
    try:
        data = request.get_json()
        
        if not data or "samples" not in data:
            return jsonify({"error": "Missing 'samples' field in request"}), 400
        
        samples = data["samples"]
        if not isinstance(samples, list):
            return jsonify({"error": "'samples' must be a list"}), 400
        
        samples_added = 0
        errors = []
        
        for i, sample in enumerate(samples):
            try:
                # Validate each sample
                required_fields = ["fft_data", "label", "session_id"]
                for field in required_fields:
                    if field not in sample:
                        errors.append(f"Sample {i}: Missing field '{field}'")
                        continue
                
                # Add sample to training data
                training_sample = {
                    "fft_data": sample["fft_data"],
                    "label": sample["label"],
                    "session_id": sample["session_id"],
                    "confidence_score": sample.get("confidence_score"),
                    "features": sample.get("features"),
                    "metadata": sample.get("metadata", {}),
                    "timestamp": datetime.now().isoformat(),
                    "sample_id": f"{sample['session_id']}_{len(training_data) + samples_added}"
                }
                
                training_data.append(training_sample)
                samples_added += 1
                
            except Exception as e:
                errors.append(f"Sample {i}: {str(e)}")
        
        logger.info(f"Batch training - Added: {samples_added}, Errors: {len(errors)}, Total: {len(training_data)}")
        
        response = {
            "message": f"Batch training completed",
            "samples_added": samples_added,
            "total_samples": len(training_data),
            "errors": errors if errors else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch training error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/training_data", methods=["GET"])
def get_training_data():
    """Get stored training data (for debugging)"""
    try:
        return jsonify({
            "total_samples": len(training_data),
            "samples": training_data[-10:] if len(training_data) > 10 else training_data,  # Last 10 samples
            "label_distribution": {
                label: sum(1 for sample in training_data if sample["label"] == label)
                for label in ["approved", "rejected", "needs_review", "pending"]
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error retrieving training data: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)