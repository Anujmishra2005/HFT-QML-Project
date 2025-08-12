"""
Helpers to encode classical features into quantum circuits.
This is illustrative. We provide simple angle encoding via PennyLane.
"""
import numpy as np

def angle_encode(features, n_qubits):
    # features -> scaled to [-pi, pi] and assigned to first n_qubits
    arr = np.array(features)
    arr = arr[:n_qubits]
    # normalize to [-pi, pi]
    if arr.max() != arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())  # [0,1]
        arr = arr * 2 * np.pi - np.pi
    else:
        arr = np.linspace(-np.pi, np.pi, n_qubits)
    return arr
