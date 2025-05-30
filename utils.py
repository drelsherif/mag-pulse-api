
import numpy as np

def preprocess_fft(data):
    data = np.array(data)
    data = data / np.max(np.abs(data))
    return data.reshape((1, len(data), 1))

def preprocess_fft_xyz(x, y, z):
    data = np.stack([x, y, z], axis=-1)
    data = data / np.max(np.abs(data))
    return data.reshape((1, len(x), 3))
