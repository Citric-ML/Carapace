# ============================================================
# BEIJING PM2.5 – RECURRENCE PLOT vs 1D CNN BASELINE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop index column
    df = df.drop(columns=["No"])

    # Create linear time index
    df["t"] = (
        df["year"] * 365 * 24 +
        df["month"] * 30 * 24 +
        df["day"] * 24 +
        df["hour"]
    )

    # Drop original time columns
    df = df.drop(columns=["year", "month", "day", "hour"])

    # One-hot encode wind direction
    df = pd.get_dummies(df, columns=["cbwd"])

    # Handle missing values
    df = df.ffill().bfill().interpolate()

    # Separate target
    target = df["pm2.5"].values

    # Normalize per feature
    features = df.columns
    for col in features:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    # Light smoothing (moving average)
    df = df.rolling(window=3, min_periods=1).mean()

    # Optional detrending
    rolling_mean = df.rolling(window=24, min_periods=1).mean()
    df = df - rolling_mean

    return df.values, target


# ============================================================
# 2. WINDOWING
# ============================================================

def create_windows(data, target, window_size=128, horizon=1):
    X, y = [], []

    for i in range(len(data) - window_size - horizon):
        X.append(data[i:i+window_size])
        y.append(target[i+window_size+horizon-1])

    return np.array(X), np.array(y)


# ============================================================
# 3. DATASETS
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 4. LEARNED EMBEDDING (1D CNN)
# ============================================================

class Embedding1DCNN(nn.Module):
    def __init__(self, in_channels, latent_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        z = self.net(x)
        return z  # (B, D, T)


# ============================================================
# 5. CONTINUOUS RECURRENCE PLOT
# ============================================================

def recurrence_plot(z):
    # z: (B, D, T) → (B, T, D)
    z = z.permute(0, 2, 1)

    # Pairwise distances
    dist = torch.cdist(z, z)  # (B, T, T)

    sigma = torch.std(dist, dim=(1,2), keepdim=True) + 1e-8

    R = torch.exp(-dist / sigma)

    # Normalize
    R_min = R.amin(dim=(1,2), keepdim=True)
    R_max = R.amax(dim=(1,2), keepdim=True)
    R = (R - R_min) / (R_max - R_min + 1e-8)

    # Resize to 64x64
    R = R.unsqueeze(1)
    R = F.interpolate(R, size=(64, 64), mode='bilinear')

    return R  # (B, 1, 64, 64)


# ============================================================
# 6. MODEL A: EMBEDDING + 2D CNN
# ============================================================
def regression_accuracy(preds, targets, epsilon=10.0):
    preds = np.array(preds)
    targets = np.array(targets)
    return np.mean(np.abs(preds - targets) < epsilon)

class RecurrenceModel(nn.Module):
    def __init__(self, in_channels, latent_dim=6):
        super().__init__()

        self.embedding = Embedding1DCNN(in_channels, latent_dim)

        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        z = self.embedding(x)
        R = recurrence_plot(z)
        out = self.cnn2d(R)
        out = self.fc(out)
        return out.squeeze(), z, R


# ============================================================
# 7. BASELINE MODEL (1D CNN)
# ============================================================

class Baseline1DCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(8)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.net(x)
        out = self.fc(out)
        return out.squeeze()


# ============================================================
# 8. TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, test_loader, epochs=10, epsilon=10.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    history = {
        "train_mse": [],
        "test_mse": [],
        "test_mae": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Noise injection
            X = X + 0.01 * torch.randn_like(X)

            optimizer.zero_grad()

            if isinstance(model, RecurrenceModel):
                pred, _, _ = model(X)
            else:
                pred = model(X)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_mse = train_loss / len(train_loader)
        test_mse, test_mae, test_acc = evaluate(model, test_loader, epsilon)

        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["test_mae"].append(test_mae)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch+1} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Test MSE: {test_mse:.4f} | "
            f"MAE: {test_mae:.4f} | "
            f"Acc: {test_acc:.4f}"
        )

    return history

# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate(model, loader, epsilon=10.0):
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            if isinstance(model, RecurrenceModel):
                pred, _, _ = model(X)
            else:
                pred = model(X)

            preds.extend(pred.cpu().numpy())
            targets.extend(y.numpy())

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    acc = regression_accuracy(preds, targets, epsilon)

    return mse, mae, acc

def full_metrics(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            if isinstance(model, RecurrenceModel):
                pred, _, _ = model(X)
            else:
                pred = model(X)

            preds.extend(pred.cpu().numpy())
            targets.extend(y.numpy())

    print("MSE:", mean_squared_error(targets, preds))
    print("MAE:", mean_absolute_error(targets, preds))
    print("R2:", r2_score(targets, preds))

    return np.array(preds), np.array(targets)

# ============================================================
# 10. VISUALIZATION
# ============================================================

def visualize_sample(model, dataset, idx=0):
    model.eval()
    X, y = dataset[idx]
    X = X.unsqueeze(0).to(device)

    with torch.no_grad():
        pred, z, R = model(X)

    X = X.cpu().squeeze().numpy()
    z = z.cpu().squeeze().numpy()
    R = R.cpu().squeeze().numpy()

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title("Raw Signals")
    plt.plot(X[:, :3])  # first 3 features

    plt.subplot(1, 3, 2)
    plt.title("Latent Trajectory")
    plt.plot(z.T)

    plt.subplot(1, 3, 3)
    plt.title("Recurrence Plot")
    plt.imshow(R, cmap='viridis')

    plt.suptitle(f"Pred: {pred.item():.2f} | Actual: {y:.2f}")
    plt.show()

def plot_training_comparison(history_A, history_B):
    epochs = range(1, len(history_A["train_mse"]) + 1)

    plt.figure(figsize=(14, 5))

    # RP Model
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_A["train_mse"], label="Train MSE")
    plt.plot(epochs, history_A["test_mse"], '--', label="Test MSE")
    plt.title("End-to-End RP Model")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalised space)")
    plt.legend()

    # Baseline
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_B["train_mse"], label="Train MSE")
    plt.plot(epochs, history_B["test_mse"], '--', label="Test MSE")
    plt.title("1D CNN Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalised space)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def compare_models(model_A, model_B, test_loader):
    preds_A, targets = full_metrics(model_A, test_loader)
    preds_B, _ = full_metrics(model_B, test_loader)

    mse_A = mean_squared_error(targets, preds_A)
    mae_A = mean_absolute_error(targets, preds_A)
    acc_A = regression_accuracy(preds_A, targets)
    r2_A = r2_score(targets, preds_A)

    mse_B = mean_squared_error(targets, preds_B)
    mae_B = mean_absolute_error(targets, preds_B)
    acc_B = regression_accuracy(preds_B, targets)
    r2_B = r2_score(targets, preds_B)

    labels = ["MSE", "MAE", "Accuracy", "R²"]

    model_A_vals = [mse_A, mae_A, acc_A, r2_A]
    model_B_vals = [mse_B, mae_B, acc_B, r2_B]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, model_A_vals, width, label="Recurrence")
    plt.bar(x + width/2, model_B_vals, width, label="Baseline")

    plt.xticks(x, labels)
    plt.title("Model Comparison")
    plt.legend()
    plt.show()

def plot_predictions_and_errors(model_A, model_B, test_loader, num_samples=300):
    model_A.eval()
    model_B.eval()

    preds_A, preds_B, targets = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            pA, _, _ = model_A(X)
            pB = model_B(X)

            preds_A.extend(pA.cpu().numpy())
            preds_B.extend(pB.cpu().numpy())
            targets.extend(y.numpy())

    preds_A = np.array(preds_A)
    preds_B = np.array(preds_B)
    targets = np.array(targets)

    errors_A = preds_A - targets
    errors_B = preds_B - targets

    mae_A = mean_absolute_error(targets, preds_A)
    mae_B = mean_absolute_error(targets, preds_B)

    plt.figure(figsize=(14, 5))

    # LEFT: predictions
    plt.subplot(1, 2, 1)
    plt.plot(targets[:num_samples], label="Actual", color="black")
    plt.plot(preds_A[:num_samples], label="RP Model", alpha=0.8)
    plt.plot(preds_B[:num_samples], '--', label="1D CNN", alpha=0.8)
    plt.title("PM2.5 Prediction — Normalised Space")
    plt.xlabel("Test sample index")
    plt.ylabel("Normalised PM2.5")
    plt.legend()

    # RIGHT: error histogram
    plt.subplot(1, 2, 2)
    plt.hist(errors_A, bins=60, alpha=0.6, label=f"RP MAE={mae_A:.4f}")
    plt.hist(errors_B, bins=60, alpha=0.6, label=f"Raw MAE={mae_B:.4f}")

    plt.axvline(0, linestyle="--", color="black")
    plt.title("Error Distribution (full test set)")
    plt.xlabel("Prediction error")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_multisample_rp(model, dataset, num_samples=4):
    model.eval()

    plt.figure(figsize=(16, 6))

    for i in range(num_samples):
        X, y = dataset[i]
        X_input = X.unsqueeze(0).to(device)

        with torch.no_grad():
            pred, z, R = model(X_input)

        X_np = X.numpy()
        R_np = R.cpu().squeeze().numpy()

        # --- TOP ROW: signals ---
        plt.subplot(2, num_samples, i + 1)
        for j in range(min(4, X_np.shape[1])):
            plt.plot(X_np[:, j], label=["DEWP", "TEMP", "PRES", "Iws"][j])
        plt.title(f"Sample {i+1}")
        plt.xlabel("Time step")

        if i == 0:
            plt.legend(fontsize=7)

        # --- BOTTOM ROW: recurrence plot ---
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(R_np, cmap='inferno')
        plt.title("Continuous RP")
        plt.axis("off")

        # Annotation
        plt.text(
            0, -10,
            f"Actual : {y:.3f}\n"
            f"RP CNN : {pred.item():.3f}",
            fontsize=9
        )

    plt.suptitle("Example Samples: Signals | Recurrence Plot | Predictions")
    plt.tight_layout()
    plt.show()

# ============================================================
# 11. MAIN EXECUTION
# ============================================================

path = "/kaggle/input/datasets/djhavera/beijing-pm25-data-data-set/PRSA_data_2010.1.1-2014.12.31.csv"

data, target = load_and_preprocess(path)

X, y = create_windows(data, target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ============================================================
# TRAIN MODELS
# ============================================================

in_channels = X.shape[2]

print("Training Recurrence Model...")
model_A = RecurrenceModel(in_channels)
history_A = train_model(model_A, train_loader, test_loader, epochs=30)

print("\nTraining Baseline Model...")
model_B = Baseline1DCNN(in_channels)
history_B = train_model(model_B, train_loader, test_loader, epochs=30)

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n=== Recurrence Model ===")
preds_A, targets_A = full_metrics(model_A, test_loader)

print("\n=== Baseline Model ===")
preds_B, targets_B = full_metrics(model_B, test_loader)

# ============================================================
# PLOTS
# ============================================================

plt.figure()
plt.scatter(targets_A, preds_A, alpha=0.3)
plt.title("Recurrence Model: Pred vs Actual")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
plt.figure()
plt.hist(targets_A - preds_A, bins=50)
plt.title("Recurrence Model Error Distribution")
plt.show()
plot_training_curves(history_A, history_B)
compare_models(model_A, model_B, test_loader)
plot_multisample_rp(model_B, pd.read_csv(path))
plot_training_comparison(history_A, history_B)
plot_predictions_and_errors(
    model_A,
    model_B,
    test_loader,
    num_samples=300  # matches your previous figure
)
plot_multisample_rp(
    model_A,
    test_dataset,
    num_samples=4  # matches your 4-panel layout
)
# ============================================================
# VISUAL SAMPLE
# ============================================================

visualize_sample(model_A, test_dataset, idx=10)
