

import numpy as np




def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot matrix."""
    n = y.shape[0]
    oh = np.zeros((n, num_classes), dtype=np.float64)
    oh[np.arange(n), y.astype(int)] = 1.0
    return oh



class CrossEntropy:
    

    def loss(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        
        probs = _softmax(logits)
        n = logits.shape[0]
        # Clip to avoid log(0)
        log_probs = np.log(np.clip(probs[np.arange(n), y_true.astype(int)], 1e-15, 1.0))
        return -np.mean(log_probs)

    def gradient(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Returns gradient of loss w.r.t. logits: (batch, C)
        """
        probs = _softmax(logits)
        oh = _one_hot(y_true, logits.shape[1])
        
        return (probs - oh)

    def __repr__(self):
        return "CrossEntropy()"


class MeanSquaredError:
    

    def loss(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        probs = _softmax(logits)
        oh = _one_hot(y_true, logits.shape[1])
        return np.mean(np.sum((probs - oh) ** 2, axis=1))

    def gradient(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        
        probs = _softmax(logits)
        oh = _one_hot(y_true, logits.shape[1])
        n = logits.shape[0]

        # (batch, C)
        diff = probs - oh  # 2* factor absorbed into grad scaling below

        
        weighted = np.sum(diff * probs, axis=1, keepdims=True)  # (batch, 1)
        
        grad = 2.0 * probs * (diff - weighted)
        return grad

    def __repr__(self):
        return "MeanSquaredError()"



LOSS_MAP = {
    "cross_entropy": CrossEntropy,
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
}


def get_loss(name: str):
    """Return an instantiated loss object."""
    name = name.lower().replace("-", "_")
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_MAP.keys())}")

    return LOSS_MAP[name]()
