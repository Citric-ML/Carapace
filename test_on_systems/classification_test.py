# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
 
# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Generating a lorenz system (testing)
def lorenz_equations(state, t, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz system ODEs.
 
    Parameters
    ----------
    state : array-like, shape (3,)
        Current [x, y, z] state.
    t     : float
        Current time (required by odeint but unused in autonomous system).
    sigma, rho, beta : float
        Standard Lorenz parameters.
 
    Returns
    -------
    [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]
 
 
def generate_lorenz(
    n_steps: int = 3000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
    transient: int = 500,
) -> np.ndarray:
    """
    Integrate the Lorenz system and return the x-component time series.
 
    Parameters
    ----------
    n_steps   : total integration steps (including transient).
    dt        : time-step size.
    transient : number of initial steps to discard (allow attractor
                to be reached before recording).
 
    Returns
    -------
    x_series : 1-D array of length (n_steps - transient).
    """
    # Random initial condition near the Lorenz attractor basin
    x0 = np.random.uniform(-15, 15, size=3)
    t  = np.linspace(0, n_steps * dt, n_steps)
 
    sol = odeint(lorenz_equations, x0, t, args=(sigma, rho, beta))
 
    # Return x-component, discard transient
    return sol[transient:, 0]

# Embed that system into phase space

def takens_embedding(
    time_series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 5,
) -> np.ndarray:
    """
    Construct a time-delay (Takens) embedding of a scalar time series.
 
    For a series x[0..N-1] with embedding dimension m and delay τ, the
    i-th embedded vector is:
 
        v_i = [x[i], x[i+τ], x[i+2τ], …, x[i+(m-1)τ]]
 
    Parameters
    ----------
    time_series   : 1-D array of length N.
    embedding_dim : number of dimensions in the reconstructed phase space.
    delay         : time-lag between successive coordinates.
 
    Returns
    -------
    embedded : 2-D array of shape (M, embedding_dim), where
               M = N - (embedding_dim - 1) * delay.
    """
    N = len(time_series)
    M = N - (embedding_dim - 1) * delay
 
    if M <= 0:
        raise ValueError(
            f"Time series (length {N}) is too short for "
            f"embedding_dim={embedding_dim}, delay={delay}."
        )
 
    embedded = np.empty((M, embedding_dim))
    for i in range(M):
        for d in range(embedding_dim):
            embedded[i, d] = time_series[i + d * delay]
 
    return embedded

# Use the trajectory data in phase space to convert to a recurrence plot
def compute_recurrence_plot(
    embedded: np.ndarray,
    epsilon: float | None = None,
    epsilon_quantile: float = 0.10,
    image_size: int = 64,
) -> np.ndarray:
    """
    Build a normalised, resized recurrence plot from a phase-space trajectory.
 
    Algorithm
    ---------
    1. Compute the N×N matrix of pairwise Euclidean distances.
    2. Threshold with ε → binary recurrence matrix R[i,j] = 1 if dist ≤ ε.
    3. Resize to (image_size × image_size) via block-averaging (no external
       imaging library required).
    4. Normalise to [0, 1].
 
    Parameters
    ----------
    embedded         : phase-space trajectory, shape (N, d).
    epsilon          : fixed distance threshold. If None, computed
                       automatically as the `epsilon_quantile` quantile of
                       all pairwise distances.
    epsilon_quantile : quantile used for automatic ε selection.
    image_size       : target spatial resolution of the output image.
 
    Returns
    -------
    rp_resized : float32 array of shape (image_size, image_size) in [0, 1].
    """
    N = len(embedded)
 
    # --- 1. Pairwise Euclidean distances ------------------------------------
    # Use broadcasting for efficiency
    diff   = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]  # (N,N,d)
    dist   = np.sqrt(np.sum(diff ** 2, axis=-1))                       # (N,N)
 
    # --- 2. Threshold -------------------------------------------------------
    if epsilon is None:
        epsilon = np.quantile(dist, epsilon_quantile)
 
    rp = (dist <= epsilon).astype(np.float32)   # binary matrix
 
    # --- 3. Resize via block-averaging (pure numpy) -------------------------
    # Crop to the largest multiple of image_size for even block division
    crop = (N // image_size) * image_size
    rp_crop = rp[:crop, :crop]
 
    block = crop // image_size
    rp_resized = (
        rp_crop
        .reshape(image_size, block, image_size, block)
        .mean(axis=(1, 3))                        # average over each block
    ).astype(np.float32)
 
    # --- 4. Normalise to [0, 1] --------------------------------------------
    mn, mx = rp_resized.min(), rp_resized.max()
    if mx > mn:
        rp_resized = (rp_resized - mn) / (mx - mn)
 
    return rp_resized
 
 
def time_series_to_rp(
    ts: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 5,
    epsilon: float | None = None,
    epsilon_quantile: float = 0.10,
    image_size: int = 64,
) -> np.ndarray:
    """Convenience wrapper: raw time series → 64×64 recurrence plot."""
    embedded = takens_embedding(ts, embedding_dim, delay)
    return compute_recurrence_plot(
        embedded,
        epsilon=epsilon,
        epsilon_quantile=epsilon_quantile,
        image_size=image_size,
    )

# Generates training/testing datasets (classifications: stable or chaotic)

# ---- Stable regime generators ----------------------------------------------
 
def generate_sine_wave(
    n_steps: int = 3000,
    dt: float = 0.01,
    transient: int = 500,
) -> np.ndarray:
    """Multi-component sine wave with random frequencies and phases."""
    t = np.linspace(0, n_steps * dt, n_steps)
    n_components = np.random.randint(1, 4)
    signal = np.zeros(n_steps)
    for _ in range(n_components):
        freq  = np.random.uniform(0.5, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amp   = np.random.uniform(0.5, 2.0)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal[transient:]
 
 
def generate_damped_oscillator(
    n_steps: int = 3000,
    dt: float = 0.01,
    transient: int = 500,
) -> np.ndarray:
    """
    Damped harmonic oscillator integrated numerically:
        x'' + 2γω₀ x' + ω₀² x = 0
    """
    omega0 = np.random.uniform(1.0, 5.0)
    gamma  = np.random.uniform(0.05, 0.3)   # under-damped
 
    def damped_osc(state, t):
        x, v = state
        dxdt = v
        dvdt = -2 * gamma * omega0 * v - omega0**2 * x
        return [dxdt, dvdt]
 
    x0 = [np.random.uniform(0.5, 2.0), 0.0]
    t  = np.linspace(0, n_steps * dt, n_steps)
    sol = odeint(damped_osc, x0, t)
    return sol[transient:, 0]
 
 
def generate_quasiperiodic(
    n_steps: int = 3000,
    dt: float = 0.01,
    transient: int = 500,
) -> np.ndarray:
    """
    Quasi-periodic signal: sum of two sinusoids with incommensurate
    frequencies (ratio ≈ golden ratio).
    """
    t  = np.linspace(0, n_steps * dt, n_steps)
    f1 = np.random.uniform(1.0, 3.0)
    f2 = f1 * (1 + np.sqrt(5)) / 2          # irrational ratio
    a1 = np.random.uniform(0.5, 1.5)
    a2 = np.random.uniform(0.5, 1.5)
    signal = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    return signal[transient:]
 
 
# ---- Dataset class ---------------------------------------------------------
 
class RecurrencePlotDataset(Dataset):
    """
    PyTorch Dataset holding (recurrence_plot, label) pairs.
 
    Labels:
        0 → stable regime
        1 → chaotic regime
    """
 
    def __init__(
        self,
        recurrence_plots: np.ndarray,  # shape (N, H, W)
        labels: np.ndarray,            # shape (N,)
        transform=None,
    ):
        self.rps       = torch.tensor(recurrence_plots, dtype=torch.float32).unsqueeze(1)
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        rp    = self.rps[idx]
        label = self.labels[idx]
        if self.transform:
            rp = self.transform(rp)
        return rp, label
 
 
def build_dataset(
    n_chaotic: int = 200,
    n_stable:  int = 200,
    n_steps:   int = 3000,
    dt:        float = 0.01,
    transient: int = 500,
    embedding_dim: int = 3,
    delay:         int = 5,
    image_size:    int = 64,
    test_size:     float = 0.20,
):
    """
    Generate recurrence plots from Lorenz (chaotic) and stable systems,
    split into train / test sets, and return DataLoaders.
 
    Returns
    -------
    train_loader, test_loader, rp_examples
        rp_examples : dict with keys 'stable' and 'chaotic', each a list of
                      raw numpy recurrence-plot arrays for visualisation.
    """
    stable_generators = [
        generate_sine_wave,
        generate_damped_oscillator,
        generate_quasiperiodic,
    ]
 
    all_rps    = []
    all_labels = []
    rp_examples = {"stable": [], "chaotic": []}
 
    print(f"Generating {n_chaotic} chaotic (Lorenz) samples …")
    for i in range(n_chaotic):
        ts = generate_lorenz(n_steps=n_steps, dt=dt, transient=transient)
        rp = time_series_to_rp(ts, embedding_dim, delay, image_size=image_size)
        all_rps.append(rp)
        all_labels.append(1)                     # 1 = chaotic
        if i < 4:
            rp_examples["chaotic"].append(rp)
 
    print(f"Generating {n_stable} stable samples …")
    for i in range(n_stable):
        gen = stable_generators[i % len(stable_generators)]
        ts  = gen(n_steps=n_steps, dt=dt, transient=transient)
        rp  = time_series_to_rp(ts, embedding_dim, delay, image_size=image_size)
        all_rps.append(rp)
        all_labels.append(0)                     # 0 = stable
        if i < 4:
            rp_examples["stable"].append(rp)
 
    all_rps    = np.array(all_rps,    dtype=np.float32)   # (N, 64, 64)
    all_labels = np.array(all_labels, dtype=np.int64)
 
    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        all_rps, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=SEED,
    )
 
    # Light data augmentation for training only (horizontal/vertical flip)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
 
    train_ds = RecurrencePlotDataset(X_train, y_train, transform=train_transform)
    test_ds  = RecurrencePlotDataset(X_test,  y_test)
 
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)
 
    print(f"Dataset ready  →  train: {len(train_ds)}  |  test: {len(test_ds)}")
    return train_loader, test_loader, rp_examples

# Builds CNN to classify regime of recurrence plot

class RecurrenceCNN(nn.Module):
    """
    Small CNN that classifies 64×64 single-channel recurrence plots.
 
    Architecture
    ------------
    Block 1:  Conv(1→16, 3×3) → BN → ReLU → MaxPool(2×2)   → 32×32
    Block 2:  Conv(16→32, 3×3) → BN → ReLU → MaxPool(2×2)  → 16×16
    Block 3:  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2×2)  →  8×8
    Block 4:  Conv(64→64, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4) → 4×4
    Classifier: Flatten → FC(1024→128) → ReLU → Dropout(0.4)
                       → FC(128→2)
 
    Output: logits for [stable, chaotic].
    """
 
    def __init__(self, n_classes: int = 2, dropout: float = 0.4):
        super().__init__()
 
        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)
 
        self.features = nn.Sequential(
            conv_block(1,  16),                          # 64→32
            conv_block(16, 32),                          # 32→16
            conv_block(32, 64),                          # 16→8
            conv_block(64, 64, pool=False),              # 8→8
            nn.AdaptiveAvgPool2d(4),                     # 8→4
        )
 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
 
 
# ---- Training & evaluation utilities ---------------------------------------
 
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total
 
 
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total
 
 
def train_model(
    model,
    train_loader,
    test_loader,
    n_epochs:   int   = 20,
    lr:         float = 1e-3,
    device:     str   = "cpu",
):
    """
    Full training loop with cosine-annealing LR schedule.
 
    Returns
    -------
    history : dict with keys 'train_loss', 'train_acc',
              'test_loss', 'test_acc'.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
 
    history = {"train_loss": [], "train_acc": [],
               "test_loss":  [], "test_acc":  []}
 
    print(f"\nTraining on {device}  |  {n_epochs} epochs\n{'─'*52}")
    print(f"{'Epoch':>6}  {'Tr Loss':>9}  {'Tr Acc':>8}  "
          f"{'Te Loss':>9}  {'Te Acc':>8}")
    print('─' * 52)
 
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
 
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
 
        print(f"{epoch:>6}  {tr_loss:>9.4f}  {tr_acc*100:>7.2f}%  "
              f"{te_loss:>9.4f}  {te_acc*100:>7.2f}%")
 
    print('─' * 52)
    print(f"Final test accuracy: {te_acc*100:.2f}%\n")
    return history

# Main loop

def visualise_examples(rp_examples: dict, save_path: str = "recurrence_plots.png"):
    """
    Display up to 4 example recurrence plots from each class side-by-side.
    """
    n_cols    = 4
    class_map = {"stable": "Stable (label 0)", "chaotic": "Chaotic (label 1)"}
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle("Example Recurrence Plots by Regime", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.4, wspace=0.25)
 
    for row, (cls, title) in enumerate(class_map.items()):
        examples = rp_examples[cls][:n_cols]
        for col, rp in enumerate(examples):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(rp, cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(f"{title}\n(sample {col+1})", fontsize=8)
            ax.axis("off")
 
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Recurrence plot figure saved → {save_path}")
    plt.show()
 
 
def visualise_training(history: dict, save_path: str = "training_curves.png"):
    """Plot loss and accuracy curves over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax1.plot(epochs, history["test_loss"],  label="Test",  linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
 
    ax2.plot(epochs, [v*100 for v in history["train_acc"]], label="Train", linewidth=2)
    ax2.plot(epochs, [v*100 for v in history["test_acc"]],  label="Test",  linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Training curves saved → {save_path}")
    plt.show()
 
 
def main():
    # ------------------------------------------------------------------ #
    # Hyper-parameters(adjust during training)                           #
    # ------------------------------------------------------------------ #
    N_CHAOTIC      = 200        # number of Lorenz samples
    N_STABLE       = 200        # number of stable samples
    N_STEPS        = 3000       # integration steps per trajectory
    DT             = 0.01       # time step
    TRANSIENT      = 500        # warm-up steps to discard
    EMBEDDING_DIM  = 3          # Takens embedding dimension
    DELAY          = 5          # time-delay τ
    IMAGE_SIZE     = 64         # recurrence plot spatial resolution
    N_EPOCHS       = 25         # training epochs
    LR             = 1e-3       # initial learning rate
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
 
    print("=" * 60)
    print("  Dynamical Regime Classifier via Recurrence Plots + CNN")
    print("=" * 60)
 
    # ------------------------------------------------------------------ #
    # Step 1: Generate trajectories, embed, and build recurrence plots   #
    # ------------------------------------------------------------------ #
    train_loader, test_loader, rp_examples = build_dataset(
        n_chaotic=N_CHAOTIC,
        n_stable=N_STABLE,
        n_steps=N_STEPS,
        dt=DT,
        transient=TRANSIENT,
        embedding_dim=EMBEDDING_DIM,
        delay=DELAY,
        image_size=IMAGE_SIZE,
    )
 
    # ------------------------------------------------------------------ #
    # Step 2: Visualise example recurrence plots                         #
    # ------------------------------------------------------------------ #
    visualise_examples(rp_examples, save_path="recurrence_plots.png")
 
    # ------------------------------------------------------------------ #
    # Step 3: Build CNN model                                            #
    # ------------------------------------------------------------------ #
    model = RecurrenceCNN(n_classes=2, dropout=0.4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: RecurrenceCNN  |  trainable parameters: {n_params:,}")
    print(model)
 
    # ------------------------------------------------------------------ #
    # Step 4: Train                                                       #
    # ------------------------------------------------------------------ #
    history = train_model(
        model,
        train_loader,
        test_loader,
        n_epochs=N_EPOCHS,
        lr=LR,
        device=DEVICE,
    )
 
    # ------------------------------------------------------------------ #
    # Step 5: Plot training curves                                        #
    # ------------------------------------------------------------------ #
    visualise_training(history, save_path="training_curves.png")
 
    # ------------------------------------------------------------------ #
    # Step 6: Final evaluation report                                     #
    # ------------------------------------------------------------------ #
    model.eval()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    _, final_acc = evaluate(model, test_loader, criterion, DEVICE)
 
    print("=" * 60)
    print(f"  FINAL TEST ACCURACY : {final_acc * 100:.2f}%")
    print("=" * 60)
 
    # Save model weights
    torch.save(model.state_dict(), "recurrence_cnn_weights.pth")
    print("Model weights saved → recurrence_cnn_weights.pth")
