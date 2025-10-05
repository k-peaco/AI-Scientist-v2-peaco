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
    experiment_data = {}

try:
    for epoch_key in experiment_data.get("epoch_tuning", {}):
        data = experiment_data["epoch_tuning"][epoch_key]

        # Plot training and validation loss
        try:
            plt.figure()
            plt.plot(data["losses"]["train"], label="Training Loss")
            plt.plot(data["losses"]["val"], label="Validation Loss")
            plt.title(f"Loss Curves: {epoch_key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{epoch_key}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting loss for {epoch_key}: {e}")
            plt.close()

        # Plot validation adoption rate
        try:
            plt.figure()
            plt.plot(data["metrics"]["val"], label="Validation Adoption Rate")
            plt.title(f"Adoption Rate Over Epochs: {epoch_key}")
            plt.xlabel("Epochs")
            plt.ylabel("Adoption Rate")
            plt.savefig(os.path.join(working_dir, f"{epoch_key}_adoption_rate.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting adoption rate for {epoch_key}: {e}")
            plt.close()
except Exception as e:
    print(f"Error processing experiment data: {e}")
