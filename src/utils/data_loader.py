"""
data_loader.py
--------------
Data Loading and Preprocessing for MNIST and Fashion-MNIST.

Provides:
    load_data(dataset, val_split, seed) -> (X_train, y_train, X_val, y_val, X_test, y_test)
    log_sample_images(wandb_run, X, y, dataset) -> logs a W&B Table with class samples
"""

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

MNIST_LABELS = [str(i) for i in range(10)]


def _load_keras_dataset(name: str):
    """Load raw arrays from keras.datasets."""
    if name == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'.")
    return X_train, y_train, X_test, y_test


def _preprocess(X: np.ndarray) -> np.ndarray:
    """Flatten (N,28,28) -> (N,784) and normalise to [0,1]."""
    return X.reshape(X.shape[0], -1).astype(np.float64) / 255.0


# ── public API ────────────────────────────────────────────────────────────────

def load_data(dataset: str = "mnist",
              val_split: float = 0.1,
              seed: int = 42):
    """
    Load and preprocess dataset.

    Parameters
    ----------
    dataset   : 'mnist' or 'fashion_mnist'
    val_split : fraction of training data to reserve for validation
    seed      : random seed for reproducible split

    Returns
    -------
    X_train, y_train : training split
    X_val,   y_val   : validation split
    X_test,  y_test  : held-out test set (never used for training)
    """
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = _load_keras_dataset(dataset)

    # Preprocess
    X_all = _preprocess(X_train_raw)
    y_all = y_train_raw.astype(np.int64)
    X_test = _preprocess(X_test_raw)
    y_test = y_test_raw.astype(np.int64)

    # Train / validation split (stratified by class)
    rng = np.random.default_rng(seed)
    n = X_all.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_split)

    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

    print(f"Loaded {dataset}: "
          f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def log_sample_images(wandb_run, X_raw, y_raw, dataset: str = "mnist",
                      samples_per_class: int = 5):
    """
    Log a W&B Table with sample images from each class (Q2.1).

    Parameters
    ----------
    wandb_run       : active W&B run
    X_raw           : (N, 28, 28) uint8 raw images (before normalisation)
    y_raw           : (N,) integer labels
    dataset         : 'mnist' or 'fashion_mnist'
    samples_per_class : number of images to log per class
    """
    import wandb

    labels = FASHION_MNIST_LABELS if dataset == "fashion_mnist" else MNIST_LABELS
    columns = ["class_id", "class_name"] + [f"sample_{i+1}" for i in range(samples_per_class)]
    table   = wandb.Table(columns=columns)

    for cls in range(10):
        idxs = np.where(y_raw == cls)[0]
        chosen = idxs[:samples_per_class]
        images = [wandb.Image(X_raw[idx]) for idx in chosen]
        # Pad if fewer samples than requested
        while len(images) < samples_per_class:
            images.append(None)
        table.add_data(cls, labels[cls], *images)

    wandb_run.log({"class_samples": table})
    print(f"Logged {samples_per_class} samples per class to W&B.")