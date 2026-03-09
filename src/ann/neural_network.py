
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class NeuralNetwork:
    

    def __init__(self, cli_args):
        self.args = cli_args
        self.layers: list = []

       
        input_dim   = 784       # MNIST / Fashion-MNIST
        num_classes = 10

        hidden_sizes = cli_args.hidden_size   # list[int]
        activation   = cli_args.activation    # str
        weight_init  = cli_args.weight_init   # str

        prev_dim = input_dim
        for h_size in hidden_sizes:
            self.layers.append(
                NeuralLayer(prev_dim, h_size,
                            activation=activation,
                            weight_init=weight_init)
            )
            prev_dim = h_size

        
        self.layers.append(
            NeuralLayer(prev_dim, num_classes,
                        activation="linear",
                        weight_init=weight_init)
        )

        
        self.loss_fn = get_loss(cli_args.loss)

        
        opt_name = cli_args.optimizer.lower()
        lr       = cli_args.learning_rate
        if opt_name in ("momentum", "nag"):
            self.optimizer = get_optimizer(opt_name, learning_rate=lr, beta=0.9)
        elif opt_name == "rmsprop":
            self.optimizer = get_optimizer(opt_name, learning_rate=lr, rho=0.9)
        elif opt_name in ("adam", "nadam"):
            self.optimizer = get_optimizer(opt_name, learning_rate=lr)
        else:
            self.optimizer = get_optimizer(opt_name, learning_rate=lr)

        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)

        # Populated during backward()
        self.grad_W = None
        self.grad_b = None

   

    def forward(self, X: np.ndarray):
        """
        Forward propagation through all layers.

        Parameters
        ----------
        X : (batch, 784)

        Returns
        -------
        logits : (batch, 10)   raw pre-softmax outputs
        probs  : (batch, 10)   softmax probabilities
        """
        a = X.astype(np.float64)
        for layer in self.layers:
            a = layer.forward(a)
        return a

   

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backpropagation through all layers.

        Parameters
        ----------
        y_true : (batch,)   integer class labels
        y_pred : (batch, C) logits from forward()

        Returns
        -------
        grad_W : object array  – index 0 = last layer, … last index = first layer
        grad_b : object array  – same ordering
        """
        
        delta = self.loss_fn.gradient(y_pred, y_true)  # (batch, C)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = None, batch_size: int = None,
              wandb_run=None, log_gradients: bool = False):
        
        epochs     = epochs     or self.args.epochs
        batch_size = batch_size or self.args.batch_size
        n          = X_train.shape[0]
        history    = {"train_loss": [], "train_acc": [],
                      "val_loss":   [], "val_acc":   []}

        use_nag = self.args.optimizer.lower() == "nag"

        for epoch in range(epochs):
            idx            = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]
            epoch_loss     = 0.0
            correct        = 0

            for start in range(0, n, batch_size):
                Xb = X_shuf[start: start + batch_size]
                yb = y_shuf[start: start + batch_size]

                if use_nag:
                    self.optimizer.apply_lookahead(self.layers)

                logits = self.forward(Xb)

                if use_nag:
                    self.optimizer.undo_lookahead(self.layers)

                batch_loss = self.loss_fn.loss(logits, yb)
                self.backward(yb, logits)
                self.optimizer.step(self.layers, weight_decay=self.weight_decay)

                epoch_loss += batch_loss * Xb.shape[0]
                correct    += int(np.sum(np.argmax(logits, axis=1) == yb))

            epoch_loss /= n
            epoch_acc   = correct / n
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_loss = self.loss_fn.loss(val_logits, y_val)
                val_acc  = float(np.mean(np.argmax(val_logits, axis=1) == y_val))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # W&B logging
            if wandb_run is not None:
                log_dict = {
                    "epoch":      epoch + 1,
                    "train_loss": epoch_loss,
                    "train_acc":  epoch_acc,
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                    log_dict["val_acc"]  = val_acc

                if log_gradients:
                    for li, layer in enumerate(self.layers[:-1]):
                        if layer.grad_W is not None:
                            log_dict[f"grad_norm_layer{li}"] = float(
                                np.linalg.norm(layer.grad_W)
                            )
                            for ni in range(min(5, layer.grad_W.shape[1])):
                                log_dict[f"grad_neuron_l{li}_n{ni}"] = float(
                                    np.linalg.norm(layer.grad_W[:, ni])
                                )
                wandb_run.log(log_dict)

            val_str = (f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                       if val_loss is not None else "")
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"train_loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}{val_str}")

        return history

   

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Compute loss and accuracy.

        Returns
        -------
        loss, accuracy, logits
        """
        logits = self.forward(X)
        loss      = self.loss_fn.loss(logits, y)
        accuracy  = float(np.mean(np.argmax(logits, axis=1) == y))
        return loss, accuracy, logits

    

    def get_weights(self) -> dict:
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def __repr__(self):
        layer_str = "\n  ".join(repr(l) for l in self.layers)

        return f"NeuralNetwork(\n  {layer_str}\n)"
