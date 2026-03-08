"""
inference.py
------------
Evaluate a trained MLP on MNIST or Fashion-MNIST test set.

Usage example:
    python inference.py --model_path best_model.npy -d fashion_mnist
                        -nhl 3 -sz 128 128 128 -a relu -w_i xavier
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_arguments():
    """Parse command-line arguments for inference (same interface as train.py)."""
    parser = argparse.ArgumentParser(
        description="Run inference / evaluation with a trained MLP"
    )

    # Data
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to evaluate on (default: fashion_mnist)")

    # Training hyper-params (needed to rebuild architecture)
    parser.add_argument("-e", "--epochs", type=int, default=15,
                        help="Epochs (used when re-creating model object, default: 15)")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                        help="Weight decay (default: 0.0)")
    parser.add_argument("-o", "--optimizer", type=str, default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimiser (default: adam)")

    # Architecture – must match saved weights
    parser.add_argument("-nhl", "--num_layers", type=int, default=3,
                        help="Number of hidden layers (default: 3)")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+",
                        default=[128, 128, 128],
                        help="Neurons per hidden layer (default: 128 128 128)")
    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Hidden layer activation (default: relu)")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier", "zeros"],
                        help="Weight initialisation (default: xavier)")

    # Loss
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse", "mean_squared_error"],
                        help="Loss function (default: cross_entropy)")

    # W&B (optional for inference too)
    parser.add_argument("-wp", "--wandb_project", type=str, default=None,
                        help="W&B project name (optional)")

    # Model path
    parser.add_argument("--model_path", type=str, default="best_model.npy",
                        help="Relative path to saved .npy weights (default: best_model.npy)")

    args = parser.parse_args()

    # Reconcile hidden_size / num_layers
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        args.hidden_size = [args.hidden_size[0]] * args.num_layers

    return args


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load model weights from a .npy file."""
    data = np.load(model_path, allow_pickle=True).item()
    return data


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a model on test data.

    Returns
    -------
    dict with keys: logits, loss, accuracy, f1, precision, recall
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    logits, _ = model.forward(X_test)
    loss      = model.loss_fn.loss(logits, y_test)
    y_pred    = np.argmax(logits, axis=1)

    accuracy  = float(np.mean(y_pred == y_test))
    f1        = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    return {
        "logits":    logits,
        "loss":      float(loss),
        "accuracy":  accuracy,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    """Main inference function."""
    args = parse_arguments()

    # Load data (we only need the test split)
    _, _, _, _, X_test, y_test = load_data(dataset=args.dataset, val_split=0.1)

    # Rebuild model architecture and load weights
    model   = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("\n=== Evaluation Results ===")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")

    print("\nEvaluation complete!")
    return results


if __name__ == "__main__":
    main()