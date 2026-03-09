
import numpy as np
from ann.activations import get_activation


class NeuralLayer:

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

        
        self.W, self.b = self._init_weights(weight_init)

        
        self.z: np.ndarray = None       
        self.a_in: np.ndarray = None    

        self.grad_W: np.ndarray = None
        self.grad_b: np.ndarray = None

    

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
        self.a_in = a_in                          
        self.z = a_in @ self.W + self.b          
        a_out = self.activation.forward(self.z)
        return a_out

    

    def backward(self, delta: np.ndarray) -> np.ndarray:
        
        d_activation = self.activation.backward(self.z)   
        delta_z = delta * d_activation                     

        batch_size = self.a_in.shape[0]

        
        self.grad_W = (self.a_in.T @ delta_z) / batch_size   
        self.grad_b = np.sum(delta_z, axis=0, keepdims=True) / batch_size  

        # Propagate error to previous layer
        delta_prev = delta_z @ self.W.T                    
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
