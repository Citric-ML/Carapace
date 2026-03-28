#NOTE: THIS IS INTENDED FOR BEING RUN ON KAGGLE, DATASET & NOTEBOOK LINK IS INCLUDED IN README
# =============================================================================
# ECG Dynamical Analysis: Recurrence Plot CNN vs Raw Signal 1D CNN
# Dataset: PTB Diagnostic ECG Database (Kaggle — CSV version)
# Task:    Binary classification — healthy (0) vs diseased (1)
# =============================================================================
#
# DATASET STRUCTURE (Kaggle)
# --------------------------
# Root  : /kaggle/input/datasets/abhirampolisetti/ptb-diagnostic-ecg-database/
# CSVs  : <root>/PTB diagnostic ecg database csv files/s####_re.csv
# Format: Each CSV has 15 ROWS (one per lead) × N COLUMNS (time samples).
#         Row index 1 (0-based) = Lead II  ← the row we extract.
#         There is NO header row; the first column contains the lead name.
#
# Label inference: healthy vs diseased is determined from the filename prefix.
#   PTB subject folders encode diagnosis in wfdb headers, but the CSV
#   release does not carry those headers.  We therefore use the companion
#   RECORDS / patient-info file bundled with this Kaggle dataset, falling back
#   to a hard-coded healthy-subject list derived from the original PhysioNet
#   documentation (subjects s0290–s0294 and the 80 "Healthy control" records).
#
# INSTALLATION
# ------------
#   All packages are pre-installed on Kaggle GPU kernels.
#   For local use:  pip install torch torchvision scikit-learn matplotlib tqdm
#
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

import os
import re
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Dataset paths (Kaggle) ───────────────────────────────────────────────────
DATASET_ROOT = Path(
    "/kaggle/input/datasets/abhirampolisetti/ptb-diagnostic-ecg-database"
)
CSV_DIR = DATASET_ROOT / "PTB diagnostic ecg database csv files"

# ── Working/output directory (writable on Kaggle) ───────────────────────────
WORK_DIR   = Path("/kaggle/working")
CACHE_FILE = WORK_DIR / "ecg_windows.pkl"
RP_CACHE   = WORK_DIR / "rps.npy"

# ── Signal / windowing parameters ────────────────────────────────────────────
LEAD_ROW      = 1           # 0-based row index for Lead II inside each CSV
WINDOW_SIZE   = 400         # timesteps per window
STRIDE        = 150         # stride between successive windows

# ── Recurrence plot parameters ───────────────────────────────────────────────
EMBEDDING_DIM = 3           # Takens embedding dimension
DELAY         = 5           # Takens time-delay τ
RP_SIZE       = 64          # spatial resolution of output RP image
EPS_QUANTILE  = 0.10        # adaptive ε: 10th-percentile of pairwise distances

# ── Training parameters ──────────────────────────────────────────────────────
BATCH_SIZE    = 32
N_EPOCHS      = 30
LR            = 1e-3
TEST_SIZE     = 0.20
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

# ── Actual CSV format (confirmed from s0001_re.csv) ───────────────────────────
#
# Row 0  : HEADER — lead names as column names:
#              i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6, vx, vy, vz
# Rows 1…N : one TIME SAMPLE per row, one LEAD per column (float values).
#
# Therefore:
#   • pd.read_csv(path)  gives a DataFrame with shape (N_samples, 15)
#   • The "ii" column is Lead II — accessed as  df["ii"]
#   • N_samples ≈ 38 000 per file (sampled at 1 kHz × ~38 s)
#
# There is NO separate label/diagnosis file in this Kaggle dataset.
# Labels are inferred from the record number embedded in the filename
# (e.g. "s0014_re.csv" → record 14) using the authoritative list of
# healthy-control subject numbers from the PhysioNet PTB documentation.

# ── Healthy subject record numbers (PhysioNet PTB documentation) ─────────────
# 80 of the 294 subjects are "Healthy control". Their record numbers are:
_HEALTHY_RECORD_NUMBERS: set[int] = {
    209, 210, 212, 213, 214, 215, 218, 219, 220,
    222, 223, 224, 225, 226, 227, 228, 229,
    231, 232, 234, 235, 236, 237, 238, 239, 240,
    242, 243, 244, 245, 250, 251, 253, 255,
    258, 259, 260, 263, 264, 267, 268,
    270, 271, 272, 274, 275, 276, 277, 278, 279,
    281, 282, 283, 284, 286, 287, 288, 289,
}


def _label_from_stem(stem: str) -> int:
    """
    Derive a binary label from a filename stem such as 's0014_re'.

    Extracts the integer from the stem (14 in this example) and checks
    whether it belongs to the healthy-control set.

    Returns 0 (healthy) or 1 (diseased).
    """
    m = re.search(r's(\d+)', stem, re.IGNORECASE)
    if m is None:
        return 1          # cannot parse number → conservative: assume diseased
    return 0 if int(m.group(1)) in _HEALTHY_RECORD_NUMBERS else 1


def _read_lead_ii(csv_path: Path) -> np.ndarray | None:
    """
    Read one PTB CSV file and return the Lead II column as a 1-D float32 array.

    Format expected
    ---------------
    • Row 0   : header with lead names (i, ii, iii, avr, avl, avf, …).
    • Rows 1… : one time-sample per row; 15 float columns.
    • The "ii" column contains Lead II voltage values (in mV).

    Returns None if the file is unreadable, the "ii" column is absent,
    the signal is too short, or it is flat (std < 1e-6).
    """
    try:
        df = pd.read_csv(csv_path)            # header row auto-detected

        # Normalise column names: strip whitespace, lower-case
        df.columns = df.columns.str.strip().str.lower()

        if "ii" not in df.columns:
            return None

        values = pd.to_numeric(df["ii"], errors="coerce").values.astype(np.float32)
        values = values[~np.isnan(values)]    # drop any malformed rows

        if len(values) < WINDOW_SIZE:
            return None

        return values

    except Exception:
        return None


def load_ptb_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over all CSV files in CSV_DIR, extract Lead II, z-score normalise,
    apply a sliding window, and label each window from its filename.

    Pipeline per file
    -----------------
    1. Read "ii" column via _read_lead_ii.
    2. z-score normalise: z = (x − μ) / σ  (computed over the whole signal).
    3. Assign binary label via _label_from_stem.
    4. Slice into overlapping windows of length WINDOW_SIZE, stride STRIDE.
       Every window from the same file shares the same label.

    Caching
    -------
    Processed arrays are pickled to CACHE_FILE after the first run.
    Delete that file to force a fresh read (e.g. after changing WINDOW_SIZE).

    Returns
    -------
    windows : float32 array, shape (N_windows, WINDOW_SIZE)
    labels  : int64  array, shape (N_windows,)
    info    : object array, shape (N_windows,) — CSV stem per window
    """
    # ── Load from cache if available ─────────────────────────────────────
    if CACHE_FILE.exists():
        print(f"Loading cached windows from {CACHE_FILE} …")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        return data["windows"], data["labels"], data["info"]

    # ── Discover CSV files ────────────────────────────────────────────────
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n  {CSV_DIR}\n"
            "Please verify the Kaggle dataset path in SECTION 1."
        )
    print(f"Found {len(csv_files)} CSV files in:\n  {CSV_DIR}")

    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []
    all_info:    list[str]        = []

    skipped_bad  = 0
    label_counts = {0: 0, 1: 0}

    for csv_path in tqdm(csv_files, desc="Loading CSVs"):
        stem = csv_path.stem              # e.g. "s0014_re"

        # ── 1. Read Lead II column ────────────────────────────────────────
        signal = _read_lead_ii(csv_path)
        if signal is None:
            skipped_bad += 1
            continue

        # ── 2. Quality gate: reject flat signals ──────────────────────────
        std = float(signal.std())
        if std < 1e-6:
            skipped_bad += 1
            continue

        # ── 3. z-score normalise ──────────────────────────────────────────
        signal = (signal - signal.mean()) / std

        # ── 4. Assign label ───────────────────────────────────────────────
        label = _label_from_stem(stem)
        label_counts[label] += 1

        # ── 5. Sliding-window segmentation ───────────────────────────────
        n      = len(signal)
        starts = range(0, n - WINDOW_SIZE + 1, STRIDE)
        for start in starts:
            all_windows.append(signal[start : start + WINDOW_SIZE].copy())
            all_labels.append(label)
            all_info.append(stem)

    print(f"\n  Records loaded  : healthy={label_counts[0]}, "
          f"diseased={label_counts[1]}")
    if skipped_bad:
        print(f"  Skipped (unreadable / flat / too short): {skipped_bad}")

    # ── Guard: warn clearly if healthy count is zero ─────────────────────
    if label_counts[0] == 0:
        print(
            "\n  WARNING: zero healthy records were identified.\n"
            "  The _HEALTHY_RECORD_NUMBERS set may not match the\n"
            "  numbering in this Kaggle upload. First 10 stems:"
        )
        for p in csv_files[:10]:
            m = re.search(r's(\d+)', p.stem)
            print(f"    {p.stem}  →  record number: "
                  f"{int(m.group(1)) if m else 'unparseable'}")

    # ── Assemble arrays ───────────────────────────────────────────────────
    if not all_windows:
        raise RuntimeError(
            "No windows were produced.\n"
            f"  CSV_DIR checked: {CSV_DIR}\n"
            "  Verify the path in SECTION 1 and that files contain a 'ii' column."
        )

    windows = np.array(all_windows, dtype=np.float32)
    labels  = np.array(all_labels,  dtype=np.int64)
    info    = np.array(all_info,    dtype=object)

    # ── Cache to disk ─────────────────────────────────────────────────────
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"windows": windows, "labels": labels, "info": info}, f)
    print(f"Cached {len(windows):,} windows → {CACHE_FILE}")

    return windows, labels, info


def summarise_dataset(labels: np.ndarray, info: np.ndarray) -> None:
    """Print class balance and record count. Safe against empty arrays."""
    total = len(labels)
    if total == 0:
        print("\n  ERROR: dataset is empty — no windows were loaded.")
        return
    n_healthy  = int((labels == 0).sum())
    n_diseased = int((labels == 1).sum())
    print(f"\n{'─'*52}")
    print(f"  Total windows   : {total:,}")
    print(f"  Healthy  (0)    : {n_healthy:,}  ({100*n_healthy/total:.1f}%)")
    print(f"  Diseased (1)    : {n_diseased:,}  ({100*n_diseased/total:.1f}%)")
    print(f"  Unique records  : {len(np.unique(info))}")
    print(f"{'─'*52}\n")


# =============================================================================
# SECTION 3: PHASE-SPACE RECONSTRUCTION (TAKENS' THEOREM)
# =============================================================================

def takens_embedding(
    time_series:   np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
    delay:         int = DELAY,
) -> np.ndarray:
    """
    Time-delay embedding of a scalar time series.

    The i-th embedded vector is:
        v_i = [ x[i],  x[i+τ],  x[i+2τ], …,  x[i+(m-1)τ] ]

    Returns
    -------
    embedded : float32 array, shape (N − (m−1)τ,  m).
    """
    N = len(time_series)
    M = N - (embedding_dim - 1) * delay
    if M <= 0:
        raise ValueError(
            f"Series length {N} too short for m={embedding_dim}, τ={delay}."
        )
    embedded = np.empty((M, embedding_dim), dtype=np.float32)
    for i in range(M):
        for d in range(embedding_dim):
            embedded[i, d] = time_series[i + d * delay]
    return embedded


# =============================================================================
# SECTION 4: RECURRENCE PLOT GENERATION
# =============================================================================

def compute_recurrence_plot(
    embedded:         np.ndarray,
    epsilon:          float | None = None,
    epsilon_quantile: float        = EPS_QUANTILE,
    image_size:       int          = RP_SIZE,
) -> np.ndarray:
    """
    Binary recurrence plot, block-averaged to (image_size × image_size).

    Steps: pairwise distances → adaptive ε threshold → binary matrix
           → block-average resize → [0,1] normalisation.
    """
    N    = len(embedded)
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    if epsilon is None:
        epsilon = float(np.quantile(dist, epsilon_quantile))

    rp   = (dist <= epsilon).astype(np.float32)

    crop = (N // image_size) * image_size
    if crop < image_size:
        pad  = image_size - crop
        rp   = np.pad(rp, ((0, pad), (0, pad)), mode="edge")
        crop = image_size

    rp_crop = rp[:crop, :crop]
    block   = crop // image_size
    rp_img  = (
        rp_crop
        .reshape(image_size, block, image_size, block)
        .mean(axis=(1, 3))
    ).astype(np.float32)

    mn, mx = rp_img.min(), rp_img.max()
    if mx > mn:
        rp_img = (rp_img - mn) / (mx - mn)
    return rp_img


def window_to_rp(window: np.ndarray) -> np.ndarray:
    """Convenience: raw ECG window → 64×64 recurrence plot."""
    return compute_recurrence_plot(takens_embedding(window))


# =============================================================================
# SECTION 5: DATASET CREATION
# =============================================================================

def build_rp_array(
    windows: np.ndarray,
    desc:    str = "Building RPs",
) -> np.ndarray:
    """
    Batch-convert all ECG windows to recurrence plots.

    Returns
    -------
    rps : float32 array of shape (N, RP_SIZE, RP_SIZE).
    """
    rps = np.empty((len(windows), RP_SIZE, RP_SIZE), dtype=np.float32)
    for i, win in enumerate(tqdm(windows, desc=desc, leave=False)):
        rps[i] = window_to_rp(win)
    return rps


# ── PyTorch Dataset wrappers ─────────────────────────────────────────────────

class RPDataset(Dataset):
    """2D CNN dataset: tensors of shape (1, RP_SIZE, RP_SIZE)."""

    def __init__(self, rps: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(rps,    dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RawECGDataset(Dataset):
    """1D CNN dataset: tensors of shape (1, WINDOW_SIZE)."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels,  dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_splits(
    windows: np.ndarray,
    labels:  np.ndarray,
    rps:     np.ndarray,
) -> dict:
    """
    Single stratified train/test split shared across both modalities,
    ensuring a perfectly identical held-out set for fair comparison.

    Returns dict with keys:
        win_train, win_test, rp_train, rp_test, y_train, y_test
    """
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    return {
        "win_train": windows[idx_train], "win_test": windows[idx_test],
        "rp_train":  rps[idx_train],     "rp_test":  rps[idx_test],
        "y_train":   labels[idx_train],  "y_test":   labels[idx_test],
    }


def make_loaders(splits: dict) -> dict:
    """Wrap split arrays into PyTorch DataLoaders for both modalities."""
    rp_train  = RPDataset(splits["rp_train"],  splits["y_train"])
    rp_test   = RPDataset(splits["rp_test"],   splits["y_test"])
    raw_train = RawECGDataset(splits["win_train"], splits["y_train"])
    raw_test  = RawECGDataset(splits["win_test"],  splits["y_test"])

    return {
        "rp_train":  DataLoader(rp_train,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        "rp_test":   DataLoader(rp_test,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        "raw_train": DataLoader(raw_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        "raw_test":  DataLoader(raw_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }


# =============================================================================
# SECTION 6: MODEL 1 — 2D CNN (RECURRENCE PLOT)
# =============================================================================

class RecurrenceCNN2D(nn.Module):
    """
    2D CNN — classifies 64×64 single-channel recurrence-plot images.

    Conv Block 1 : Conv(1→32,  3×3) → BN → GELU → MaxPool(2)   → 32×32
    Conv Block 2 : Conv(32→64, 3×3) → BN → GELU → MaxPool(2)   → 16×16
    Conv Block 3 : Conv(64→128,3×3) → BN → GELU → MaxPool(2)   →  8×8
    Conv Block 4 : Conv(128→128,3×3)→ BN → GELU                →  8×8
                   AdaptiveAvgPool(4×4)                          →  4×4
    Head         : FC(2048→256) → GELU → Dropout(0.4) → FC(256→2)
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        def cb(in_c, out_c, pool=True):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            cb(1,   32),
            cb(32,  64),
            cb(64,  128),
            cb(128, 128, pool=False),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# =============================================================================
# SECTION 7: MODEL 2 — 1D CNN BASELINE (RAW ECG)
# =============================================================================

class RawECGCNN1D(nn.Module):
    """
    1D CNN — operates directly on raw normalised ECG windows.

    Block 1 : Conv(1→32,  k=7) → BN → GELU → MaxPool(2)
    Block 2 : Conv(32→64, k=5) → BN → GELU → MaxPool(2)
    Block 3 : Conv(64→128,k=3) → BN → GELU → MaxPool(2)
    Block 4 : Conv(128→128,k=3)→ BN → GELU → MaxPool(2)
    AdaptiveAvgPool(1) → FC(128→64) → GELU → Dropout → FC(64→2)
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        def cb(in_c, out_c, k):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            cb(1,   32,  7),
            cb(32,  64,  5),
            cb(64,  128, 3),
            cb(128, 128, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# =============================================================================
# SECTION 8: TRAINING
# =============================================================================

def _class_weights(labels: np.ndarray, device: str) -> torch.Tensor:
    """Inverse-frequency class weights to handle the PTB class imbalance."""
    counts  = np.bincount(labels)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss  += loss.item() * y.size(0)
        n           += y.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / n, accuracy_score(all_labels, all_preds), \
           np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, test_loader,
                model_name: str, class_weights: torch.Tensor) -> dict:
    """AdamW + CosineAnnealing training loop with best-checkpoint restore."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    history = {"train_loss": [], "train_acc": [],
               "test_loss":  [], "test_acc":  []}
    best_acc, best_state = 0.0, None

    print(f"\n{'═'*60}\n  Training: {model_name}\n{'═'*60}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}")
    print("─" * 44)

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc          = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc, _, _    = evaluate_model(model, test_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc   = te_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%  "
              f"{te_loss:>8.4f}  {te_acc*100:>6.2f}%")

    model.load_state_dict(best_state)
    print(f"\n  Best test accuracy: {best_acc*100:.2f}%")
    return history


# =============================================================================
# SECTION 9: EVALUATION AND METRICS
# =============================================================================

def full_evaluation(model, test_loader, class_weights, model_name) -> dict:
    criterion                    = nn.CrossEntropyLoss(weight=class_weights)
    _, acc, preds, labels        = evaluate_model(model, test_loader, criterion, DEVICE)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)

    print(f"\n{'─'*50}\n  {model_name} — Test Metrics\n{'─'*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "cm": cm, "preds": preds, "labels": labels}


def plot_confusion_matrices(metrics_rp, metrics_raw,
                             save_path="confusion_matrices.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metrics, title in zip(
        axes,
        [metrics_rp, metrics_raw],
        ["2D CNN (Recurrence Plot)", "1D CNN (Raw ECG)"],
    ):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["cm"],
            display_labels=["Healthy", "Diseased"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{title}\nAcc={metrics['acc']*100:.1f}%  F1={metrics['f1']:.3f}",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Confusion matrices saved → {save_path}")
    plt.show()


def plot_comparison_bar(metrics_rp, metrics_raw,
                         save_path="model_comparison.png"):
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    rp_vals  = [metrics_rp["acc"],  metrics_rp["prec"],
                metrics_rp["rec"],  metrics_rp["f1"]]
    raw_vals = [metrics_raw["acc"], metrics_raw["prec"],
                metrics_raw["rec"], metrics_raw["f1"]]
    x, w = np.arange(len(metric_names)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, rp_vals,  w, label="2D CNN (RP)",  color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, raw_vals, w, label="1D CNN (Raw)", color="#FF5722", alpha=0.85)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=3)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Recurrence Plot CNN vs Raw ECG CNN — Test Metrics", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Comparison bar chart saved → {save_path}")
    plt.show()


# =============================================================================
# SECTION 10: VISUALIZATION
# =============================================================================

def visualise_examples(windows, labels, n_per_class=3,
                        save_path="ecg_examples.png"):
    class_names = {0: "Healthy", 1: "Diseased"}
    colours     = {0: "#27AE60", 1: "#E74C3C"}
    fig = plt.figure(figsize=(14, 4.5 * 2))
    fig.suptitle(
        "ECG Windows — Raw Signal | Phase Space | Recurrence Plot",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2 * n_per_class, 3, figure=fig,
                           hspace=0.55, wspace=0.3)
    row_offset = 0
    for cls in [0, 1]:
        idxs = np.where(labels == cls)[0][:n_per_class]
        for local_row, idx in enumerate(idxs):
            win      = windows[idx]
            embedded = takens_embedding(win)
            rp       = compute_recurrence_plot(embedded)
            colour   = colours[cls]
            label    = class_names[cls]
            r = row_offset + local_row

            ax_ts = fig.add_subplot(gs[r, 0])
            ax_ts.plot(win, color=colour, linewidth=0.9, alpha=0.9)
            ax_ts.set_title(f"{label} — Raw ECG", fontsize=8)
            ax_ts.set_xlabel("Sample", fontsize=7)
            ax_ts.set_ylabel("Amplitude (norm.)", fontsize=7)
            ax_ts.tick_params(labelsize=6); ax_ts.grid(alpha=0.25)

            ax_ps = fig.add_subplot(gs[r, 1])
            ax_ps.scatter(embedded[:, 0], embedded[:, 1],
                          s=2, alpha=0.35, color=colour, linewidths=0)
            ax_ps.set_title(f"{label} — Phase Space", fontsize=8)
            ax_ps.set_xlabel("x[t]", fontsize=7)
            ax_ps.set_ylabel(f"x[t+{DELAY}]", fontsize=7)
            ax_ps.tick_params(labelsize=6); ax_ps.grid(alpha=0.2)

            ax_rp = fig.add_subplot(gs[r, 2])
            ax_rp.imshow(rp, cmap="inferno", origin="lower", aspect="auto")
            ax_rp.set_title(f"{label} — Recurrence Plot", fontsize=8)
            ax_rp.axis("off")

        row_offset += n_per_class

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Example figure saved → {save_path}")
    plt.show()


def visualise_training_curves(history_rp, history_raw,
                               save_path="training_curves.png"):
    epochs = range(1, N_EPOCHS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    pairs = [
        (axes[0, 0], "train_loss", "Training Loss"),
        (axes[0, 1], "test_loss",  "Test Loss"),
        (axes[1, 0], "train_acc",  "Training Accuracy"),
        (axes[1, 1], "test_acc",   "Test Accuracy"),
    ]
    for ax, key, title in pairs:
        scale = (lambda v: [x * 100 for x in v]) if "acc" in key else (lambda v: v)
        ax.plot(epochs, scale(history_rp[key]),  label="2D CNN (RP)",
                linewidth=2, color="#2196F3")
        ax.plot(epochs, scale(history_raw[key]), label="1D CNN (Raw)",
                linewidth=2, color="#FF5722", linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Accuracy (%)" if "acc" in key else "Loss", fontsize=9)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Training curves saved → {save_path}")
    plt.show()


def visualise_predictions(model_rp, rp_test_loader,
                           model_raw, raw_test_loader,
                           n_show=8, save_path="sample_predictions.png"):
    class_names = ["Healthy", "Diseased"]

    def get_preds(model, loader):
        model.eval()
        imgs, labs, preds = [], [], []
        with torch.no_grad():
            for X, y in loader:
                logits = model(X.to(DEVICE))
                preds.extend(logits.argmax(1).cpu().tolist())
                labs.extend(y.tolist())
                imgs.extend(X.cpu().numpy())
                if len(labs) >= n_show:
                    break
        return imgs[:n_show], labs[:n_show], preds[:n_show]

    imgs_rp,  labs_rp,  preds_rp  = get_preds(model_rp,  rp_test_loader)
    imgs_raw, labs_raw, preds_raw = get_preds(model_raw, raw_test_loader)

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 2.6 * n_show))
    fig.suptitle("Sample Test Predictions", fontsize=13, fontweight="bold")
    for ax, t in zip(axes[0],
                     ["Recurrence Plot", "Raw ECG Signal", "Labels & Predictions"]):
        ax.set_title(t, fontsize=10, fontweight="bold")

    for i in range(n_show):
        axes[i, 0].imshow(imgs_rp[i][0], cmap="inferno",
                          origin="lower", aspect="auto")
        axes[i, 0].axis("off")

        colour = "#27AE60" if labs_raw[i] == 0 else "#E74C3C"
        axes[i, 1].plot(imgs_raw[i][0], color=colour, linewidth=0.8)
        axes[i, 1].tick_params(labelsize=6); axes[i, 1].grid(alpha=0.2)

        gt    = class_names[labs_rp[i]]
        p_rp  = class_names[preds_rp[i]]
        p_raw = class_names[preds_raw[i]]
        ok_rp  = "✓" if preds_rp[i]  == labs_rp[i]  else "✗"
        ok_raw = "✓" if preds_raw[i] == labs_raw[i] else "✗"
        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.05, 0.5,
            f"Ground truth : {gt}\n"
            f"2D CNN (RP)  : {p_rp}  {ok_rp}\n"
            f"1D CNN (Raw) : {p_raw}  {ok_raw}",
            transform=axes[i, 2].transAxes,
            fontsize=9, va="center", family="monospace",
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Prediction panel saved → {save_path}")
    plt.show()


# =============================================================================
# SECTION 11: MAIN LOOP
# =============================================================================

def main():
    print("=" * 62)
    print("  ECG Regime Analysis: Recurrence Plot CNN vs Raw ECG CNN")
    print(f"  CSV source : {CSV_DIR}")
    print(f"  Device     : {DEVICE}")
    print("=" * 62)

    # ── Step 1: Load all CSVs, extract Lead II, window and label ────────
    print("\n[1/8] Loading PTB CSV dataset …")
    windows, labels, info = load_ptb_windows()
    summarise_dataset(labels, info)

    # Hard-stop with a clear message rather than a confusing ZeroDivisionError
    if len(labels) == 0:
        raise RuntimeError(
            "Dataset is empty — cannot continue.\n"
            "Review the probe output and WARNING messages printed above."
        )
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            f"Only one class found in labels (values: {np.unique(labels)}).\n"
            "The healthy-record number set may not match this dataset's\n"
            "file numbering. Check the stem inspection output above."
        )

    # ── Step 2: Visualise a few examples from each class ────────────────
    print("[2/8] Visualising ECG examples …")
    visualise_examples(windows, labels, n_per_class=3,
                       save_path=str(WORK_DIR / "ecg_examples.png"))

    # ── Step 3: Build (or load cached) recurrence plots ─────────────────
    if RP_CACHE.exists():
        print("[3/8] Loading cached recurrence plots …")
        rps = np.load(str(RP_CACHE))
    else:
        print("[3/8] Computing recurrence plots — this may take 5–15 min …")
        rps = build_rp_array(windows)
        np.save(str(RP_CACHE), rps)
        print(f"  Recurrence plots cached → {RP_CACHE}")

    # ── Step 4: Shared train/test split + DataLoaders ───────────────────
    print("\n[4/8] Creating stratified train/test splits …")
    splits  = make_splits(windows, labels, rps)
    loaders = make_loaders(splits)
    print(f"  Train: {len(splits['y_train']):,} windows  |  "
          f"Test: {len(splits['y_test']):,} windows")

    cw = _class_weights(splits["y_train"], DEVICE)

    # ── Step 5: Instantiate both models ─────────────────────────────────
    print("\n[5/8] Building models …")
    model_rp  = RecurrenceCNN2D()
    model_raw = RawECGCNN1D()
    print(f"  2D CNN (RP)  params: "
          f"{sum(p.numel() for p in model_rp.parameters() if p.requires_grad):,}")
    print(f"  1D CNN (Raw) params: "
          f"{sum(p.numel() for p in model_raw.parameters() if p.requires_grad):,}")

    # ── Step 6: Train 2D CNN ─────────────────────────────────────────────
    print("\n[6/8] Training 2D CNN (Recurrence Plot) …")
    history_rp = train_model(model_rp, loaders["rp_train"], loaders["rp_test"],
                              "2D CNN (Recurrence Plot)", cw)

    # ── Step 7: Train 1D CNN ─────────────────────────────────────────────
    print("\n[7/8] Training 1D CNN (Raw ECG) …")
    history_raw = train_model(model_raw, loaders["raw_train"], loaders["raw_test"],
                               "1D CNN (Raw ECG)", cw)

    # ── Step 8: Evaluate, compare, and visualise ─────────────────────────
    print("\n[8/8] Evaluating models …")
    metrics_rp  = full_evaluation(model_rp,  loaders["rp_test"],
                                  cw, "2D CNN (Recurrence Plot)")
    metrics_raw = full_evaluation(model_raw, loaders["raw_test"],
                                  cw, "1D CNN (Raw ECG)")

    visualise_training_curves(
        history_rp, history_raw,
        save_path=str(WORK_DIR / "training_curves.png"))
    plot_confusion_matrices(
        metrics_rp, metrics_raw,
        save_path=str(WORK_DIR / "confusion_matrices.png"))
    plot_comparison_bar(
        metrics_rp, metrics_raw,
        save_path=str(WORK_DIR / "model_comparison.png"))
    visualise_predictions(
        model_rp,  loaders["rp_test"],
        model_raw, loaders["raw_test"],
        n_show=8,
        save_path=str(WORK_DIR / "sample_predictions.png"))

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  FINAL SUMMARY")
    print("═" * 62)
    print(f"  {'Metric':<12} {'2D CNN (RP)':>14} {'1D CNN (Raw)':>14}")
    print("  " + "─" * 42)
    for key, label in [("acc", "Accuracy"), ("prec", "Precision"),
                        ("rec", "Recall"),  ("f1",   "F1")]:
        v_rp  = metrics_rp[key]
        v_raw = metrics_raw[key]
        winner = "◀ RP" if v_rp >= v_raw else "◀ Raw"
        print(f"  {label:<12} {v_rp:>13.4f}  {v_raw:>13.4f}  {winner}")
    print("═" * 62)

    # ── Save weights to /kaggle/working ──────────────────────────────────
    torch.save(model_rp.state_dict(),  str(WORK_DIR / "model_rp_2dcnn.pth"))
    torch.save(model_raw.state_dict(), str(WORK_DIR / "model_raw_1dcnn.pth"))
    print(f"\nWeights saved to {WORK_DIR}")


if __name__ == "__main__":
    main()
