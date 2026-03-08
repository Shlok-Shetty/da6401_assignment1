"""
test.py
-------
Autograder-compatible test script.
Loads best_model.npy, runs a forward pass on random data, and prints F1 score.
"""

import os
import sys
import numpy as np
import argparse
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork

# ── Best model configuration (update after actual training) ──────────────────
best_config = argparse.Namespace(
    dataset="mnist",
    epochs=15,
    batch_size=64,
    loss="cross_entropy",
    optimizer="adam",
    weight_decay=0.0,
    learning_rate=1e-3,
    num_layers=3,
    hidden_size=[128, 128, 128],
    activation="relu",
    weight_init="xavier",
)

# ── Load model and evaluate ───────────────────────────────────────────────────
model = NeuralNetwork(best_config)
weights = np.load("best_model.npy", allow_pickle=True).item()
model.set_weights(weights)

X_test = np.random.rand(100, 784)          # 100 samples, 784 features
y_true = np.random.randint(0, 10, size=(100,))  # random labels for smoke-test

y_pred = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print("F1 Score:", f1_score(y_true, y_pred_labels, average="macro", zero_division=0))