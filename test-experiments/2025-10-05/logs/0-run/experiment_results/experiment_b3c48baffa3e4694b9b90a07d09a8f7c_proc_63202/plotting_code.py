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

# Plot training and validation loss curve
try:
    plt.figure()
    train_loss = experiment_data["gamification_dataset"]["losses"]["train"]
    val_loss = experiment_data["gamification_dataset"]["losses"]["val"]
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "gamification_dataset_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot validation adoption rate curve
try:
    plt.figure()
    val_metrics = experiment_data["gamification_dataset"]["metrics"]["val"]
    plt.plot(val_metrics, label="Validation Adoption Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Adoption Rate")
    plt.title("Validation Adoption Rate over Epochs")
    plt.savefig(os.path.join(working_dir, "gamification_dataset_adoption_rate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating adoption rate plot: {e}")
    plt.close()
