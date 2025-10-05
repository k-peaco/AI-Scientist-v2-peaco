import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "ablation_activation_function": {
        "relu_activation": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic data
np.random.seed(42)
num_samples = 1000
age = np.random.randint(18, 60, size=num_samples)
gamification_points = np.random.randint(0, 100, size=num_samples)
leaderboard_position = np.random.randint(1, 101, size=num_samples)
reward_level = np.random.choice([0, 1, 2], size=num_samples, p=[0.5, 0.3, 0.2])
adopted = (
    0.5 * gamification_points
    + 0.3 * (100 - leaderboard_position)
    + 0.2 * reward_level * 50
    + np.random.normal(0, 5, num_samples)
) > 50
adopted = adopted.astype(int)

# Normalize features and split dataset
features = np.stack(
    [age, gamification_points, leaderboard_position, reward_level], axis=1
)
features = (features - features.mean(axis=0)) / features.std(axis=0)
labels = adopted
dataset = TensorDataset(
    torch.tensor(features, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32),
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define ablation model with ReLU activation
class ReLUClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ReLUClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.relu(self.fc(x))


# Training function
def train_relu_model(num_epochs, key):
    model = ReLUClassifier(input_dim=4).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in zip(["features", "labels"], batch)}
            predictions = model(batch["features"]).squeeze()
            loss = criterion(predictions, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        experiment_data["ablation_activation_function"][key]["losses"]["train"].append(
            train_loss
        )

        # Validation
        model.eval()
        val_loss, val_predictions, val_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in zip(["features", "labels"], batch)}
                predictions = model(batch["features"]).squeeze()
                loss = criterion(predictions, batch["labels"])
                val_loss += loss.item()
                val_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                val_labels.extend(batch["labels"].cpu().numpy())
        val_loss /= len(val_loader)
        experiment_data["ablation_activation_function"][key]["losses"]["val"].append(
            val_loss
        )

        # Calculate adoption rate
        val_predictions_binary = np.array(val_predictions) > 0.5
        adoption_rate = np.mean(val_predictions_binary == val_labels)
        experiment_data["ablation_activation_function"][key]["metrics"]["val"].append(
            adoption_rate
        )

        print(
            f"Epoch {epoch + 1} ({key}): train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, adoption_rate = {adoption_rate:.4f}"
        )

    experiment_data["ablation_activation_function"][key][
        "predictions"
    ] = val_predictions
    experiment_data["ablation_activation_function"][key]["ground_truth"] = val_labels


# Train ReLU model
train_relu_model(20, "relu_activation")

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Plot results
plt.figure(figsize=(12, 5))
results = experiment_data["ablation_activation_function"]["relu_activation"]
plt.subplot(1, 2, 1)
plt.plot(results["losses"]["train"], label="Train Loss")
plt.plot(results["losses"]["val"], label="Validation Loss")
plt.legend()
plt.title("Loss over Epochs: ReLU Activation")
plt.subplot(1, 2, 2)
plt.plot(results["metrics"]["val"])
plt.title("Adoption Rate: ReLU Activation")
plt.ylabel("Adoption Rate")
plt.xlabel("Epoch")

plt.tight_layout()
plt.savefig(os.path.join(working_dir, "relu_training_results.png"))
plt.show()
