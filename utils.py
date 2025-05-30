import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_fft(data):
    """
    Preprocess FFT data for model input
    
    Args:
        data: List or array of FFT values
        
    Returns:
        Preprocessed numpy array suitable for model input
    """
    try:
        # Convert to numpy array
        data = np.array(data, dtype=float)
        
        # Handle edge cases
        if len(data) == 0:
            raise ValueError("Empty FFT data provided")
            
        # Check for invalid values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("FFT data contains NaN or Inf values, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize data
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        else:
            logger.warning("FFT data is all zeros, keeping as is")
        
        # Return in format expected by models
        # For LSTM: (batch_size, timesteps, features) = (1, len(data), 1)
        # For sklearn models: will be flattened later
        return data.reshape((1, len(data), 1))
        
    except Exception as e:
        logger.error(f"Error preprocessing FFT data: {e}")
        raise ValueError(f"Failed to preprocess FFT data: {e}")

def preprocess_fft_xyz(x, y, z):
    """
    Preprocess multi-axis FFT data
    
    Args:
        x, y, z: Lists or arrays of FFT values for each axis
        
    Returns:
        Preprocessed numpy array with shape (1, timesteps, 3)
    """
    try:
        # Convert to numpy arrays
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        z = np.array(z, dtype=float)
        
        # Validate lengths match
        if not (len(x) == len(y) == len(z)):
            raise ValueError("X, Y, Z arrays must have the same length")
            
        if len(x) == 0:
            raise ValueError("Empty axis data provided")
        
        # Stack along last axis
        data = np.stack([x, y, z], axis=-1)  # Shape: (timesteps, 3)
        
        # Handle invalid values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Multi-axis FFT data contains NaN or Inf values, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize data
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        else:
            logger.warning("Multi-axis FFT data is all zeros, keeping as is")
        
        # Add batch dimension: (1, timesteps, 3)
        return data.reshape((1, len(x), 3))
        
    except Exception as e:
        logger.error(f"Error preprocessing multi-axis FFT data: {e}")
        raise ValueError(f"Failed to preprocess multi-axis FFT data: {e}")

def extract_features(fft_data):
    """
    Extract statistical features from FFT data
    
    Args:
        fft_data: Array of FFT values
        
    Returns:
        Array of extracted features
    """
    try:
        data = np.array(fft_data, dtype=float)
        
        if len(data) == 0:
            return np.array([])
        
        # Handle invalid values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(data),
            np.std(data),
            np.var(data),
            np.min(data),
            np.max(data),
            np.median(data)
        ])
        
        # Percentiles
        features.extend([
            np.percentile(data, 25),
            np.percentile(data, 75),
            np.percentile(data, 90),
            np.percentile(data, 95)
        ])
        
        # Frequency domain features
        if len(data) > 1:
            # Peak frequency
            peak_idx = np.argmax(np.abs(data))
            features.append(peak_idx / len(data))  # Normalized peak frequency
            
            # Spectral centroid
            freqs = np.arange(len(data))
            spectral_centroid = np.sum(freqs * np.abs(data)) / np.sum(np.abs(data))
            features.append(spectral_centroid / len(data))  # Normalized
            
            # Spectral rolloff (90% of energy)
            cumsum = np.cumsum(np.abs(data))
            rolloff_idx = np.where(cumsum >= 0.9 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                features.append(rolloff_idx[0] / len(data))
            else:
                features.append(1.0)
                
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Energy measures
        features.extend([
            np.sum(np.abs(data)),  # Total energy
            np.sum(data**2),       # Power
            np.sqrt(np.mean(data**2))  # RMS
        ])
        
        # Zero crossing rate (for time domain interpretation)
        if len(data) > 1:
            zero_crossings = np.sum(np.diff(np.signbit(data)))
            features.append(zero_crossings / (len(data) - 1))
        else:
            features.append(0.0)
        
        return np.array(features, dtype=float)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return np.array([0.0] * 17)  # Return zeros if feature extraction fails

def validate_model_input(data, expected_shape=None):
    """
    Validate input data for model prediction
    
    Args:
        data: Input data array
        expected_shape: Optional expected shape tuple
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid
    """
    try:
        if data is None:
            return False, "Data is None"
        
        if not isinstance(data, np.ndarray):
            return False, "Data must be numpy array"
        
        if data.size == 0:
            return False, "Data is empty"
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Data contains NaN or Inf values"
        
        if expected_shape and data.shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {data.shape}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"