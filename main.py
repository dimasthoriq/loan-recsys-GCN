import matplotlib.pyplot as plt
import torch
from utils import get_sparse_matrices
from trainers import Trainer

config = {
    # Training
    "epochs": 1000,
    "learning_rate": 5e-2,
    "weight_decay": 1e-6,
    "patience": 50,
    "min_delta": 1e-6,
    "sched_factor": 0.5,
    "sched_patience": 10,

    # Model
    "layers": 3,
    "embed_dim": 32,
    "weighted": False,

    # Data
    "path": './prosperLoanData.csv',
    "discrete_levels": 9,
    "train_size": 0.8,

    # Utils
    "seed": 15,
    "print_every": 10,
    "val_every": 25,
    "save_path": './Experiments/'
}

if __name__ == '__main__':
    R, P, Q = get_sparse_matrices(config["path"], config["discrete_levels"])
    for i in [False, True]:
        config["weighted"] = i
        if i:
            config['learning_rate'] = 5e-3
        print("\nWeighted: ", i)
        for seed in [15, 24, 35]:
            print("\nSeed: ", seed)
            config["seed"] = seed
            trainer = Trainer(R, P, Q, config)
            train_losses, val_losses = trainer.train()
            model = trainer.model

            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train loss')
            plt.plot(val_losses, label='Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(str(config["save_path"]) + str(i) + "W_" + str(seed) + "_loss" + ".png")
