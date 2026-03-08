"""
neural_layer.py
---------------
A single fully-connected (dense) layer.

Attributes stored after forward():
    self.z   : pre-activation  (batch, out_features)
    self.a_in: input activations from previous layer (batch, in_features)

Attributes stored after backward():
    self.grad_W : gradient w.r.t. W  (in_features, out_features)
    self.grad_b : gradient w.r.t. b  (1, out_features)
"""

import numpy as np
from ann.activations import get_activation


class NeuralLayer:
    """
    One dense layer: z = a_in @ W + b,  a_out = activation(z)

    Parameters
    ----------
    in_features  : int   - number of input neurons
    out_features : int   - number of output neurons
    activation   : str   - activation function name ('sigmoid','tanh','relu','linear')
    weight_init  : str   - 'random' or 'xavier'
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        weight_init: str = "xavier",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.activation = get_activation(activation)

        # Initialise weights
        self.W, self.b = self._init_weights(weight_init)

        # Placeholders populated during forward / backward
        self.z: np.ndarray = None       # pre-activation
        self.a_in: np.ndarray = None    # cached input

        self.grad_W: np.ndarray = None
        self.grad_b: np.ndarray = None

    # ── weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self, method: str):
        method = method.lower()
        if method == "zeros":
            W = np.zeros((self.in_features, self.out_features))
            b = np.zeros((1, self.out_features))
        elif method == "random":
            W = np.random.randn(self.in_features, self.out_features) * 0.01
            b = np.zeros((1, self.out_features))
        elif method == "xavier":
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            W = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
            b = np.zeros((1, self.out_features))
        else:
            raise ValueError(
                f"Unknown weight_init '{method}'. Choose from 'random', 'xavier', 'zeros'."
            )
        return W.astype(np.float64), b.astype(np.float64)

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, a_in: np.ndarray) -> np.ndarray:
        """
        Compute layer output.

        Parameters
        ----------
        a_in : (batch, in_features)

        Returns
        -------
        a_out : (batch, out_features)  - activated output
        """
        self.a_in = a_in                          # cache for backward
        self.z = a_in @ self.W + self.b           # (batch, out_features)
        a_out = self.activation.forward(self.z)
        return a_out

    # ── backward pass ─────────────────────────────────────────────────────────

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate error signal through this layer.

        Parameters
        ----------
        delta : (batch, out_features)
            Error signal arriving at the OUTPUT of this layer's activation.
            For the last (logit) layer, delta = dL/dz (already multiplied by
            activation derivative outside).  For hidden layers this method
            multiplies by the activation derivative internally.

        Returns
        -------
        delta_prev : (batch, in_features)
            Error signal to propagate to the previous layer.
        """
        # Multiply by local activation derivative
        d_activation = self.activation.backward(self.z)   # (batch, out_features)
        delta_z = delta * d_activation                     # element-wise

        batch_size = self.a_in.shape[0]

        # Parameter gradients (averaged over batch)
        self.grad_W = (self.a_in.T @ delta_z) / batch_size   # (in, out)
        self.grad_b = np.sum(delta_z, axis=0, keepdims=True) / batch_size  # (1, out)

        # Propagate error to previous layer
        delta_prev = delta_z @ self.W.T                    # (batch, in_features)
        return delta_prev


    def get_weights(self) -> dict:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weights: dict):
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()

    def __repr__(self):
        return (
            f"NeuralLayer(in={self.in_features}, out={self.out_features}, "
            f"activation={self.activation_name})"
        )