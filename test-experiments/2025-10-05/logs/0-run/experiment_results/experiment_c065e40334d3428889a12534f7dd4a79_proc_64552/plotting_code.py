import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: Training and Validation Loss
try:
    plt.figure()
    losses = experiment_data["ablation_activation_function"]["relu_activation"][
        "losses"
    ]
    plt.plot(losses["train"], label="Train Loss")
    plt.plot(losses["val"], label="Validation Loss")
    plt.title("Loss Curves (ReLU Activation)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "relu_activation_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 2: Validation Adoption Rate
try:
    plt.figure()
    adoption_rate = experiment_data["ablation_activation_function"]["relu_activation"][
        "metrics"
    ]["val"]
    plt.plot(adoption_rate)
    plt.title("Validation Adoption Rate (ReLU Activation)")
    plt.xlabel("Epochs")
    plt.ylabel("Adoption Rate")
    plt.savefig(os.path.join(working_dir, "relu_activation_adoption_rate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating adoption rate plot: {e}")
    plt.close()
