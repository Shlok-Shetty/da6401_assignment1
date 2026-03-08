
import numpy as np


class Sigmoid:
    """Sigmoid activation: σ(z) = 1 / (1 + exp(-z))"""

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def backward(self, z: np.ndarray) -> np.ndarray:
        s = self.forward(z)
        return s * (1.0 - s)

    def __repr__(self):
        return "Sigmoid()"


class Tanh:
    """Hyperbolic tangent activation: tanh(z)"""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2

    def __repr__(self):
        return "Tanh()"


class ReLU:
    """Rectified Linear Unit: max(0, z)"""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def __repr__(self):
        return "ReLU()"


class Linear:
    """Identity / linear activation (used for output layer logits)."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)

    def __repr__(self):
        return "Linear()"




ACTIVATION_MAP = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "linear": Linear,
}


def get_activation(name: str):
    """Return an instantiated activation object for the given name string."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(ACTIVATION_MAP.keys())}"
        )
    return ACTIVATION_MAP[name]()