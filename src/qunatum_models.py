"""
Minimal VQC example using PennyLane. This file is illustrative and includes a simulator example.
To run real quantum experiments, configure devices/backends and manage runtime carefully.
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import joblib
from pathlib import Path
from .config import QUANTUM_DIR

n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

def variational_circuit(params, x):
    # x: array of angle encodings
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)
    # variational layer
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

@qml.qnode(dev, interface='autograd')
def vqc_node(params, x):
    return variational_circuit(params, x)

def vqc_train(X, y, steps=30):
    # X shape (N, n_qubits), y in {-1,0,1} -> we will map to {-1,1} for binary demo
    # simple demo: convert to binary by mapping non-zero to 1
    y_bin = np.where(y == 0, -1, 1)
    params = pnp.random.randn(1, n_qubits, 3)  # shape matching StronglyEntanglingLayers
    opt = qml.AdamOptimizer(0.1)
    for i in range(steps):
        def cost(p):
            preds = [np.sum(vqc_node(p, x)) for x in X]
            return np.mean((preds - y_bin) ** 2)
        params = opt.step(cost, params)
        if i % 10 == 0:
            print("Step", i)
    # save params
    Path(QUANTUM_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(params, Path(QUANTUM_DIR) / "vqc_params.pkl")
    return params
