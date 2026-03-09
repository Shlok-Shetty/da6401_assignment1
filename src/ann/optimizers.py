
import numpy as np




class BaseOptimizer:
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def step(self, layers: list, weight_decay: float = 0.0):
        raise NotImplementedError

    def _apply_weight_decay(self, layers, weight_decay):
        """L2 regularisation: add wd * W to gradient."""
        if weight_decay > 0:
            for layer in layers:
                layer.grad_W = layer.grad_W + weight_decay * layer.W
        return layers



class SGD(BaseOptimizer):
    """
    Mini-batch Stochastic Gradient Descent.
        W <- W - lr * (dL/dW + wd * W)
        b <- b - lr * dL/db
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, layers: list, weight_decay: float = 0.0):
        self._apply_weight_decay(layers, weight_decay)
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

    def __repr__(self):
        return f"SGD(lr={self.lr})"




class Momentum(BaseOptimizer):
    """
    SGD with classical momentum.
        v_W <- beta * v_W + dL/dW
        W   <- W - lr * v_W
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W: dict = {}
        self.v_b: dict = {}

    def _init_velocities(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def step(self, layers: list, weight_decay: float = 0.0):
        self._init_velocities(layers)
        self._apply_weight_decay(layers, weight_decay)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]

    def __repr__(self):
        return f"Momentum(lr={self.lr}, beta={self.beta})"




class NAG(BaseOptimizer):
    

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W: dict = {}
        self.v_b: dict = {}

    def _init_velocities(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.v_W:
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def apply_lookahead(self, layers: list):
        
        self._init_velocities(layers)
        for i, layer in enumerate(layers):
            layer.W -= self.lr * self.beta * self.v_W[i]
            layer.b -= self.lr * self.beta * self.v_b[i]

    def undo_lookahead(self, layers: list):
        
        for i, layer in enumerate(layers):
            layer.W += self.lr * self.beta * self.v_W[i]
            layer.b += self.lr * self.beta * self.v_b[i]

    def step(self, layers: list, weight_decay: float = 0.0):
        self._init_velocities(layers)
        self._apply_weight_decay(layers, weight_decay)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]

    def __repr__(self):
        return f"NAG(lr={self.lr}, beta={self.beta})"




class RMSProp(BaseOptimizer):
   
    def __init__(self, learning_rate: float = 0.001,
                 rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho     = rho
        self.epsilon = epsilon
        self.s_W: dict = {}
        self.s_b: dict = {}

    def _init_cache(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.s_W:
                self.s_W[i] = np.zeros_like(layer.W)
                self.s_b[i] = np.zeros_like(layer.b)

    def step(self, layers: list, weight_decay: float = 0.0):
        self._init_cache(layers)
        self._apply_weight_decay(layers, weight_decay)
        for i, layer in enumerate(layers):
            self.s_W[i] = (self.rho * self.s_W[i] +
                           (1 - self.rho) * layer.grad_W ** 2)
            self.s_b[i] = (self.rho * self.s_b[i] +
                           (1 - self.rho) * layer.grad_b ** 2)
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.epsilon)

    def __repr__(self):
        return f"RMSProp(lr={self.lr}, rho={self.rho})"



class Adam(BaseOptimizer):
    

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.m_W: dict = {}
        self.m_b: dict = {}
        self.v_W: dict = {}
        self.v_b: dict = {}
        self.t = 0

    def _init_moments(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.m_W:
                self.m_W[i] = np.zeros_like(layer.W)
                self.m_b[i] = np.zeros_like(layer.b)
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def step(self, layers: list, weight_decay: float = 0.0):
        self._init_moments(layers)
        self._apply_weight_decay(layers, weight_decay)
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self.t) /
                          (1 - self.beta1 ** self.t))
        for i, layer in enumerate(layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * layer.grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * layer.grad_b ** 2
            layer.W -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.epsilon)
            layer.b -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"




class Nadam(BaseOptimizer):
   

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.m_W: dict = {}
        self.m_b: dict = {}
        self.v_W: dict = {}
        self.v_b: dict = {}
        self.t = 0

    def _init_moments(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.m_W:
                self.m_W[i] = np.zeros_like(layer.W)
                self.m_b[i] = np.zeros_like(layer.b)
                self.v_W[i] = np.zeros_like(layer.W)
                self.v_b[i] = np.zeros_like(layer.b)

    def step(self, layers: list, weight_decay: float = 0.0):
        self._init_moments(layers)
        self._apply_weight_decay(layers, weight_decay)
        self.t += 1
        for i, layer in enumerate(layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * layer.grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * layer.grad_b ** 2

            m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            
            nesterov_W = (self.beta1 * m_hat_W +
                          (1 - self.beta1) / (1 - self.beta1 ** self.t) * layer.grad_W)
            nesterov_b = (self.beta1 * m_hat_b +
                          (1 - self.beta1) / (1 - self.beta1 ** self.t) * layer.grad_b)

            layer.W -= self.lr * nesterov_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.b -= self.lr * nesterov_b / (np.sqrt(v_hat_b) + self.epsilon)

    def __repr__(self):
        return f"Nadam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"



OPTIMIZER_MAP = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}


def get_optimizer(name: str, learning_rate: float, **kwargs):
    """Return an instantiated optimizer by name."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_MAP.keys())}"
        )

    return OPTIMIZER_MAP[name](learning_rate=learning_rate, **kwargs)
