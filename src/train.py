

import argparse
import json
import os
import sys
import numpy as np

# Allow running from the src/ directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, log_sample_images




def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP on MNIST or Fashion-MNIST"
    )

    # Data
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use (default: fashion_mnist)")

    # Training
    parser.add_argument("-e", "--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                        help="L2 weight decay coefficient (default: 0.0)")

    # Optimiser
    parser.add_argument("-o", "--optimizer", type=str, default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimiser (default: adam)")

    # Architecture
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

    # W&B
    parser.add_argument("-wp", "--wandb_project", type=str, default=None,
                        help="Weights & Biases project name (skip W&B if not provided)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity / username")
    parser.add_argument("--log_gradients", action="store_true",
                        help="Log per-layer gradient norms to W&B")

    # Model saving
    parser.add_argument("--model_save_path", type=str, default="best_model.npy",
                        help="Relative path to save best model weights (default: best_model.npy)")
    parser.add_argument("--config_save_path", type=str, default="best_config.json",
                        help="Relative path to save best config JSON (default: best_config.json)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation fraction from training set (default: 0.1)")
    parser.add_argument("--log_class_samples", action="store_true",
                        help="Log class sample images to W&B (Q2.1)")

    args = parser.parse_args()

    # Validate hidden_size length matches num_layers
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        print(f"[WARNING] --hidden_size has {len(args.hidden_size)} values but "
              f"--num_layers={args.num_layers}. Using first value repeated.")
        args.hidden_size = [args.hidden_size[0]] * args.num_layers

    return args




def set_seed(seed: int):
    np.random.seed(seed)


def save_model(model: NeuralNetwork, config, model_path: str, config_path: str):
    """Serialise model weights (.npy) and hyperconfig (.json)."""
    weights = model.get_weights()
    np.save(model_path, weights)

    cfg = {
        "dataset":      config.dataset,
        "epochs":       config.epochs,
        "batch_size":   config.batch_size,
        "loss":         config.loss,
        "optimizer":    config.optimizer,
        "weight_decay": config.weight_decay,
        "learning_rate": config.learning_rate,
        "num_layers":   config.num_layers,
        "hidden_size":  config.hidden_size,
        "activation":   config.activation,
        "weight_init":  config.weight_init,
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Model saved to '{model_path}', config to '{config_path}'")




def main():
    args = parse_arguments()
    set_seed(args.seed)

    
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                reinit=True,
            )
        except Exception as e:
            print(f"[WARNING] Could not initialise W&B: {e}")

   
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        dataset=args.dataset, val_split=args.val_split, seed=args.seed
    )

    
    if wandb_run and args.log_class_samples:
        from keras.datasets import mnist, fashion_mnist
        if args.dataset == "mnist":
            (X_raw, y_raw), _ = mnist.load_data()
        else:
            (X_raw, y_raw), _ = fashion_mnist.load_data()
        log_sample_images(wandb_run, X_raw, y_raw, dataset=args.dataset)

    
    model = NeuralNetwork(args)
    print(model)

    
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        wandb_run=wandb_run,
        log_gradients=args.log_gradients,
    )

    
    from sklearn.metrics import f1_score, precision_score, recall_score

    test_loss, test_acc, test_logits = model.evaluate(X_test, y_test)
    y_pred_labels = np.argmax(test_logits, axis=1)

    f1        = f1_score(y_test, y_pred_labels, average="macro", zero_division=0)
    precision = precision_score(y_test, y_pred_labels, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred_labels, average="macro", zero_division=0)

    print("\n=== Test Results ===")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    if wandb_run:
        wandb_run.log({
            "test_loss":      test_loss,
            "test_acc":       test_acc,
            "test_f1":        f1,
            "test_precision": precision,
            "test_recall":    recall,
        })

    
    save_model(model, args, args.model_save_path, args.config_save_path)

    if wandb_run:
        wandb_run.finish()

    print("\nTraining complete!")
    return history


if __name__ == "__main__":

    main()
