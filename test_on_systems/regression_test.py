# =============================================================================
# SECTION 1: IMPORT STATEMENTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# r range for the logistic map
R_MIN = 2.5
R_MAX = 4.0


# =============================================================================
# SECTION 2: LOGISTIC MAP GENERATION
# =============================================================================

def generate_logistic_map(
    r: float,
    n_steps: int = 400,
    transient: int = 100,
    x0: float | None = None,
) -> np.ndarray:
    """
    Iterate the logistic map  x_{t+1} = r · x_t · (1 - x_t)  and return the
    post-transient time series.

    Parameters
    ----------
    r         : control parameter in [2.5, 4.0].
    n_steps   : total iterations (including transient).
    transient : number of initial steps to discard before recording.
    x0        : initial condition in (0, 1); drawn randomly if None.

    Returns
    -------
    series : 1-D float32 array of length (n_steps - transient).

    Notes
    -----
    The logistic map is a discrete map on (0, 1):
      - r < 3.0  → convergence to a fixed point
      - r ∈ [3.0, 3.57) → period-doubling cascade
      - r ≳ 3.57 → onset of chaos
      - r = 4.0  → fully developed chaos
    Discarding the transient ensures the orbit has reached its attractor
    before we start recording.
    """
    if x0 is None:
        x0 = np.random.uniform(0.1, 0.9)

    x      = np.empty(n_steps, dtype=np.float64)
    x[0]   = x0
    for t in range(1, n_steps):
        x[t] = r * x[t - 1] * (1.0 - x[t - 1])

    return x[transient:].astype(np.float32)


def normalise_r(r: float, r_min: float = R_MIN, r_max: float = R_MAX) -> float:
    """Linearly map r ∈ [r_min, r_max] → [0, 1] for stable training targets."""
    return (r - r_min) / (r_max - r_min)


def denormalise_r(r_norm: float, r_min: float = R_MIN, r_max: float = R_MAX) -> float:
    """Invert normalise_r: [0, 1] → [r_min, r_max]."""
    return r_norm * (r_max - r_min) + r_min


# =============================================================================
# SECTION 3: PHASE-SPACE EMBEDDING (TAKENS' THEOREM)
# =============================================================================

def takens_embedding(
    time_series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 5,
) -> np.ndarray:
    """
    Construct a time-delay (Takens) embedding of a scalar time series.

    For a series x[0..N-1] with embedding dimension m and delay τ, the
    i-th embedded vector is:

        v_i = [ x[i],  x[i+τ],  x[i+2τ], …,  x[i+(m-1)τ] ]

    Parameters
    ----------
    time_series   : 1-D array of length N.
    embedding_dim : number of reconstructed phase-space dimensions.
    delay         : time-lag τ between successive coordinates.

    Returns
    -------
    embedded : 2-D float32 array of shape (M, embedding_dim), where
               M = N - (embedding_dim - 1) * delay.
    """
    N = len(time_series)
    M = N - (embedding_dim - 1) * delay

    if M <= 0:
        raise ValueError(
            f"Time series (length {N}) too short for "
            f"embedding_dim={embedding_dim}, delay={delay}. "
            f"Need at least {(embedding_dim - 1) * delay + 1} points."
        )

    embedded = np.empty((M, embedding_dim), dtype=np.float32)
    for i in range(M):
        for d in range(embedding_dim):
            embedded[i, d] = time_series[i + d * delay]

    return embedded


# =============================================================================
# SECTION 4: RECURRENCE PLOT CONSTRUCTION
# =============================================================================

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
    2. Threshold with ε  →  binary recurrence matrix  R[i,j] = 1 if dist ≤ ε.
    3. Resize to (image_size × image_size) via block-averaging (pure NumPy).
    4. Normalise to [0, 1].

    Parameters
    ----------
    embedded         : phase-space trajectory, shape (N, d).
    epsilon          : fixed distance threshold. If None, set automatically
                       as the `epsilon_quantile` quantile of all pairwise
                       distances — this makes ε scale-invariant across
                       different r values.
    epsilon_quantile : quantile used for automatic ε selection.
    image_size       : target spatial resolution.

    Returns
    -------
    rp_resized : float32 array of shape (image_size, image_size) in [0, 1].
    """
    N = len(embedded)

    # 1. Pairwise Euclidean distances (broadcasting)
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]  # (N,N,d)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))                       # (N,N)

    # 2. Threshold → binary recurrence matrix
    if epsilon is None:
        epsilon = np.quantile(dist, epsilon_quantile)
    rp = (dist <= epsilon).astype(np.float32)

    # 3. Resize via block-averaging
    #    Crop to largest multiple of image_size so blocks divide evenly.
    crop     = (N // image_size) * image_size
    if crop < image_size:
        raise ValueError(
            f"Embedded trajectory ({N} points) is too short to form a "
            f"{image_size}×{image_size} recurrence plot after cropping."
        )
    rp_crop  = rp[:crop, :crop]
    block    = crop // image_size
    rp_resized = (
        rp_crop
        .reshape(image_size, block, image_size, block)
        .mean(axis=(1, 3))
    ).astype(np.float32)

    # 4. Normalise to [0, 1]
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
    """Convenience wrapper: raw time series → (image_size × image_size) RP."""
    embedded = takens_embedding(ts, embedding_dim, delay)
    return compute_recurrence_plot(
        embedded,
        epsilon=epsilon,
        epsilon_quantile=epsilon_quantile,
        image_size=image_size,
    )


# =============================================================================
# SECTION 5: DATASET CREATION
# =============================================================================

class LogisticMapRPDataset(Dataset):
    """
    PyTorch Dataset of (recurrence_plot, normalised_r) pairs.

    Attributes
    ----------
    rps      : tensor of shape (N, 1, H, W)  — single-channel images.
    r_norms  : tensor of shape (N, 1)         — regression targets in [0, 1].
    r_raw    : numpy array of shape (N,)      — original r values (for display).
    """

    def __init__(
        self,
        rps:     np.ndarray,    # (N, H, W)  float32
        r_norms: np.ndarray,    # (N,)       float32  normalised r
        r_raw:   np.ndarray,    # (N,)       float32  original r
        transform=None,
    ):
        self.rps     = torch.tensor(rps,     dtype=torch.float32).unsqueeze(1)
        self.r_norms = torch.tensor(r_norms, dtype=torch.float32).unsqueeze(1)
        self.r_raw   = r_raw
        self.transform = transform

    def __len__(self):
        return len(self.r_norms)

    def __getitem__(self, idx):
        rp     = self.rps[idx]
        target = self.r_norms[idx]
        if self.transform:
            rp = self.transform(rp)
        return rp, target


def build_dataset(
    n_samples:     int   = 800,
    n_steps:       int   = 400,
    transient:     int   = 100,
    embedding_dim: int   = 3,
    delay:         int   = 5,
    image_size:    int   = 64,
    test_size:     float = 0.20,
    r_min:         float = R_MIN,
    r_max:         float = R_MAX,
):
    """
    Sample r uniformly in [r_min, r_max], generate logistic map time series,
    embed via Takens' theorem, convert to recurrence plots, and return
    train/test DataLoaders plus raw example data for visualisation.

    Returns
    -------
    train_loader, test_loader, examples
        examples : list of dicts, each with keys
                   'ts'  (time series), 'embedded' (phase trajectory),
                   'rp'  (recurrence plot), 'r' (true r value).
    """
    r_values = np.random.uniform(r_min, r_max, size=n_samples).astype(np.float32)

    all_rps    = []
    all_rnorms = []
    examples   = []   # keep a few for plotting

    print(f"Generating {n_samples} logistic-map recurrence plots …")
    for i, r in enumerate(r_values):
        ts       = generate_logistic_map(r, n_steps=n_steps, transient=transient)
        rp       = time_series_to_rp(ts, embedding_dim, delay, image_size=image_size)
        r_norm   = normalise_r(r, r_min, r_max)

        all_rps.append(rp)
        all_rnorms.append(r_norm)

        # Store first 8 examples (covering different r regimes)
        if i < 8:
            embedded = takens_embedding(ts, embedding_dim, delay)
            examples.append({"ts": ts, "embedded": embedded, "rp": rp, "r": float(r)})

        if (i + 1) % 200 == 0:
            print(f"  … {i + 1}/{n_samples} done")

    all_rps    = np.array(all_rps,    dtype=np.float32)   # (N, 64, 64)
    all_rnorms = np.array(all_rnorms, dtype=np.float32)   # (N,)

    # Stratified split is not applicable for regression; use random split
    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
        all_rps, all_rnorms, r_values,
        test_size=test_size,
        random_state=SEED,
    )

    # Light augmentation: random flips (RP symmetry preserves recurrence info)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    train_ds = LogisticMapRPDataset(X_train, y_train, r_train, transform=train_transform)
    test_ds  = LogisticMapRPDataset(X_test,  y_test,  r_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    print(f"Dataset ready  →  train: {len(train_ds)}  |  test: {len(test_ds)}\n")
    return train_loader, test_loader, examples


# =============================================================================
# SECTION 6: VISUALISATION
# =============================================================================

def visualise_examples(examples: list, save_path: str = "logistic_examples.png"):
    """
    For each example in `examples`, plot three panels side-by-side:
      (A) raw time series  |  (B) 2-D phase-space projection  |  (C) recurrence plot

    Parameters
    ----------
    examples  : list of dicts from build_dataset, each containing
                'ts', 'embedded', 'rp', 'r'.
    save_path : output filename.
    """
    n       = len(examples)
    fig     = plt.figure(figsize=(13, 3.2 * n))
    fig.suptitle(
        "Logistic Map Examples: Time Series | Phase Space | Recurrence Plot",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.35)

    cmap_ts  = plt.cm.plasma
    cmap_rp  = "inferno"

    for row, ex in enumerate(examples):
        r        = ex["r"]
        ts       = ex["ts"]
        embedded = ex["embedded"]
        rp       = ex["rp"]
        t_idx    = np.arange(len(ts))

        # ── (A) Time series ─────────────────────────────────────────────
        ax_ts = fig.add_subplot(gs[row, 0])
        ax_ts.plot(t_idx[:150], ts[:150], linewidth=0.9,
                   color=cmap_ts(normalise_r(r)), alpha=0.85)
        ax_ts.set_title(f"r = {r:.4f}  —  Time Series", fontsize=8)
        ax_ts.set_xlabel("t", fontsize=7)
        ax_ts.set_ylabel("x_t", fontsize=7)
        ax_ts.tick_params(labelsize=6)
        ax_ts.set_ylim(-0.05, 1.05)
        ax_ts.grid(alpha=0.25)

        # ── (B) Phase-space projection (x_t vs x_{t+τ}) ─────────────────
        ax_ps = fig.add_subplot(gs[row, 1])
        ax_ps.scatter(embedded[:, 0], embedded[:, 1],
                      s=2, alpha=0.4, linewidths=0,
                      color=cmap_ts(normalise_r(r)))
        ax_ps.set_title(f"r = {r:.4f}  —  Phase Space (d=1,2)", fontsize=8)
        ax_ps.set_xlabel("x[t]", fontsize=7)
        ax_ps.set_ylabel(f"x[t+{5}]", fontsize=7)
        ax_ps.tick_params(labelsize=6)
        ax_ps.grid(alpha=0.2)

        # ── (C) Recurrence plot ──────────────────────────────────────────
        ax_rp = fig.add_subplot(gs[row, 2])
        ax_rp.imshow(rp, cmap=cmap_rp, origin="lower", aspect="auto",
                     vmin=0, vmax=1)
        ax_rp.set_title(f"r = {r:.4f}  —  Recurrence Plot", fontsize=8)
        ax_rp.axis("off")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Example figure saved → {save_path}")
    plt.show()


def visualise_training(history: dict, save_path: str = "regression_training.png"):
    """Plot MSE loss curves over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, history["train_loss"], label="Train MSE",  linewidth=2)
    ax.plot(epochs, history["test_loss"],  label="Test MSE",   linewidth=2,
            linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss (normalised r)", fontsize=11)
    ax.set_title("Regression Training — MSE Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Training-curve figure saved → {save_path}")
    plt.show()


def visualise_predictions(
    model,
    test_loader,
    device: str,
    n_show: int = 40,
    save_path: str = "predictions.png",
):
    """
    Scatter-plot predicted r vs true r for the first `n_show` test samples,
    and draw the ideal y = x diagonal.
    """
    model.eval()
    true_list, pred_list = [], []

    with torch.no_grad():
        for images, targets in test_loader:
            images  = images.to(device)
            preds   = model(images).squeeze(1).cpu().numpy()
            targets = targets.squeeze(1).numpy()
            true_list.extend(targets.tolist())
            pred_list.extend(preds.tolist())
            if len(true_list) >= n_show:
                break

    true_r = np.array([denormalise_r(v) for v in true_list[:n_show]])
    pred_r = np.array([denormalise_r(v) for v in pred_list[:n_show]])

    mae_r = np.mean(np.abs(true_r - pred_r))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_r, pred_r, alpha=0.7, s=45, edgecolors="none",
               c=true_r, cmap="plasma")
    lims = [R_MIN - 0.05, R_MAX + 0.05]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True r", fontsize=12)
    ax.set_ylabel("Predicted r", fontsize=12)
    ax.set_title(
        f"Predicted vs True r  (first {n_show} test samples)\n"
        f"MAE = {mae_r:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Prediction-scatter figure saved → {save_path}")
    plt.show()


# =============================================================================
# SECTION 6 (cont.): CNN FOR REGRESSION
# =============================================================================

class RecurrenceRegressionCNN(nn.Module):
    """
    CNN that predicts the logistic-map parameter r from a 64×64
    single-channel recurrence plot.

    Key differences from the classification variant
    ------------------------------------------------
    * Output head:  Linear(128 → 1)  — single continuous output neuron.
    * No activation on the output layer — raw logit is the prediction.
    * Loss function (used externally): MSELoss on the normalised target.

    Architecture
    ------------
    Block 1:  Conv(1→16,  3×3) → BN → GELU → MaxPool(2×2)          32×32
    Block 2:  Conv(16→32, 3×3) → BN → GELU → MaxPool(2×2)          16×16
    Block 3:  Conv(32→64, 3×3) → BN → GELU → MaxPool(2×2)           8×8
    Block 4:  Conv(64→64, 3×3) → BN → GELU                          8×8
              AdaptiveAvgPool(4×4)                                    4×4
    Head:     Flatten → FC(1024→256) → GELU → Dropout(0.3)
                      → FC(256→64)  → GELU
                      → FC(64→1)                   ← no activation
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,  16),          # 64→32
            conv_block(16, 32),          # 32→16
            conv_block(32, 64),          # 16→8
            conv_block(64, 64, pool=False),
            nn.AdaptiveAvgPool2d(4),     # 8→4
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),            # ← single output, no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))


# =============================================================================
# SECTION 7: TRAINING AND EVALUATION
# =============================================================================

def train_one_epoch_regression(model, loader, criterion, optimizer, device):
    """One forward/backward pass over the training DataLoader."""
    model.train()
    total_loss, total = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, targets)
        loss.backward()
        # Gradient clipping to guard against instability early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total      += targets.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate_regression(model, loader, criterion, device):
    """
    Evaluate on a DataLoader.

    Returns
    -------
    mse_norm : MSE in normalised-r space (training signal).
    mse_r    : MSE in original r units.
    mae_r    : MAE in original r units.
    """
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss  = criterion(preds, targets)
        total_loss  += loss.item() * targets.size(0)
        total       += targets.size(0)
        all_preds.append(preds.squeeze(1).cpu().numpy())
        all_targets.append(targets.squeeze(1).cpu().numpy())

    mse_norm = total_loss / total

    # Convert back to original r scale for interpretable metrics
    preds_r   = np.concatenate(all_preds)   * (R_MAX - R_MIN) + R_MIN
    targets_r = np.concatenate(all_targets) * (R_MAX - R_MIN) + R_MIN
    mse_r     = float(np.mean((preds_r - targets_r) ** 2))
    mae_r     = float(np.mean(np.abs(preds_r - targets_r)))

    return mse_norm, mse_r, mae_r


def train_model(
    model,
    train_loader,
    test_loader,
    n_epochs: int   = 40,
    lr:       float = 1e-3,
    device:   str   = "cpu",
):
    """
    Full regression training loop.

    Optimiser : AdamW with weight decay.
    Schedule  : CosineAnnealingLR.
    Loss      : MSELoss on normalised r.

    Returns
    -------
    history : dict with keys 'train_loss', 'test_loss'.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    history = {"train_loss": [], "test_loss": []}

    print(f"Training on {device}  |  {n_epochs} epochs")
    print(f"{'─'*64}")
    print(f"{'Epoch':>6}  {'Tr MSE(norm)':>13}  {'Te MSE(norm)':>13}  "
          f"{'Te MSE(r)':>10}  {'Te MAE(r)':>10}")
    print('─' * 64)

    for epoch in range(1, n_epochs + 1):
        tr_loss              = train_one_epoch_regression(
                                   model, train_loader, criterion, optimizer, device)
        te_loss, te_mse, te_mae = evaluate_regression(
                                   model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)

        print(f"{epoch:>6}  {tr_loss:>13.6f}  {te_loss:>13.6f}  "
              f"{te_mse:>10.6f}  {te_mae:>10.6f}")

    print('─' * 64)
    return history


# =============================================================================
# SECTION 8: MAIN LOOP
# =============================================================================

def main():
    # ------------------------------------------------------------------ #
    # Hyper-parameters                                                    #
    # ------------------------------------------------------------------ #
    N_SAMPLES      = 1000       # total logistic-map samples
    N_STEPS        = 500        # total iterations per trajectory
    TRANSIENT      = 100        # warm-up iterations to discard
    EMBEDDING_DIM  = 3          # Takens embedding dimension
    DELAY          = 5          # time-delay τ
    IMAGE_SIZE     = 64         # recurrence plot spatial resolution
    N_EPOCHS       = 40         # training epochs
    LR             = 1e-3       # peak learning rate
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print("  Logistic Map r-Parameter Regression via Recurrence CNN")
    print("=" * 64, "\n")

    # ------------------------------------------------------------------ #
    # Step 1: Generate dataset                                            #
    # ------------------------------------------------------------------ #
    train_loader, test_loader, examples = build_dataset(
        n_samples=N_SAMPLES,
        n_steps=N_STEPS,
        transient=TRANSIENT,
        embedding_dim=EMBEDDING_DIM,
        delay=DELAY,
        image_size=IMAGE_SIZE,
    )

    # ------------------------------------------------------------------ #
    # Step 2: Visualise examples (time series, phase space, RP)          #
    # ------------------------------------------------------------------ #
    # Sort examples by r so the figure shows a clear progression
    examples_sorted = sorted(examples, key=lambda e: e["r"])
    visualise_examples(examples_sorted, save_path="logistic_examples.png")

    # ------------------------------------------------------------------ #
    # Step 3: Build regression CNN                                        #
    # ------------------------------------------------------------------ #
    model    = RecurrenceRegressionCNN(dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: RecurrenceRegressionCNN  |  trainable parameters: {n_params:,}")
    print(model, "\n")

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
    visualise_training(history, save_path="regression_training.png")

    # ------------------------------------------------------------------ #
    # Step 6: Final evaluation — MSE and MAE on the full test set        #
    # ------------------------------------------------------------------ #
    model.eval()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    _, final_mse, final_mae = evaluate_regression(model, test_loader, criterion, DEVICE)

    print("\n" + "=" * 64)
    print(f"  FINAL TEST MSE (r units) : {final_mse:.6f}")
    print(f"  FINAL TEST MAE (r units) : {final_mae:.6f}")
    print(f"  (r range [{R_MIN}, {R_MAX}]  →  baseline MAE ≈ {(R_MAX - R_MIN)/4:.4f})")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # Step 7: Predicted vs true r scatter plot                           #
    # ------------------------------------------------------------------ #
    visualise_predictions(
        model, test_loader, DEVICE,
        n_show=min(80, len(test_loader.dataset)),
        save_path="predictions.png",
    )

    # ------------------------------------------------------------------ #
    # Step 8: Example predictions printed to console                     #
    # ------------------------------------------------------------------ #
    print("\nSample predictions (first 10 test examples):")
    print(f"  {'True r':>10}  {'Pred r':>10}  {'|Error|':>10}")
    print("  " + "─" * 34)
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images  = images.to(DEVICE)
            preds   = model(images).squeeze(1).cpu().numpy()
            targets = targets.squeeze(1).numpy()
            for true_n, pred_n in zip(targets[:10], preds[:10]):
                true_r = denormalise_r(true_n)
                pred_r = denormalise_r(pred_n)
                print(f"  {true_r:>10.4f}  {pred_r:>10.4f}  {abs(true_r-pred_r):>10.4f}")
            break

    # Save model weights
    torch.save(model.state_dict(), "logistic_regression_cnn.pth")
    print("\nModel weights saved → logistic_regression_cnn.pth")


if __name__ == "__main__":
    main()
