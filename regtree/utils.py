import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def neighbor_avg(array: np.ndarray) -> np.ndarray:
    array = np.unique(array)
    array.sort()
    return np.vstack([array[1:], array[:-1]]).mean(axis=0)

def mse(value: np.float64, array: np.ndarray) -> np.float64:
    return np.square(array - value).mean()

