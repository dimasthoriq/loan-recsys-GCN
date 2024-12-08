import matplotlib.pyplot as plt
import torch
from utils import get_sparse_matrices
from trainers import Trainer

config = {
    # Training
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 15,
    "min_delta": 1e-4,
    "sched_factor": 0.5,
    "sched_patience": 5,

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
    "print_every": 5,
    "val_every": 10,
    "save_path": './Experiments/'
}

R, P, Q = get_sparse_matrices(config["path"], config["discrete_levels"])
trainer = Trainer(R, P, Q, config)
train_losses, val_losses = trainer.train()
model = trainer.model

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(str(config["save_path"]) + 'loss.png')
