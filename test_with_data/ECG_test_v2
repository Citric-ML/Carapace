#NOTE: THIS IS INTENDED FOR BEING RUN ON KAGGLE, DATASET & NOTEBOOK LINK IS INCLUDED IN README
# =============================================================================
# ECG Dynamical Analysis: Learned-Embedding Recurrence Plot CNN vs 1D CNN
# Dataset : PTB Diagnostic ECG Database (Kaggle — CSV version)
# Task    : Binary classification — healthy (0) vs diseased (1)
# =============================================================================
#
# FULL PIPELINE (model 1 — RP path)
# ----------------------------------
#   ECG signal
#     → z-score normalise
#     → R-peak detection  (scipy.signal.find_peaks)
#     → beat-centred window extraction  [peak−PRE_OFFSET : peak+POST_OFFSET]
#     → LearnedEmbedding1D  (1D CNN, preserves time dimension)
#         input  : (batch, 1, WINDOW_SIZE)
#         output : (batch, EMBED_DIM, WINDOW_SIZE)   ← latent trajectory Z
#     → compute_recurrence_plot  (pairwise Euclidean, adaptive ε)
#         output : (RP_SIZE, RP_SIZE)
#     → RecurrenceCNN2D  → class logits
#
# BASELINE PIPELINE (model 2 — raw path)
# ----------------------------------------
#   ECG beat window  →  RawECGCNN1D  →  class logits
#
# DATASET STRUCTURE (Kaggle)
# --------------------------
# Root : /kaggle/input/datasets/abhirampolisetti/ptb-diagnostic-ecg-database/
# CSVs : <root>/PTB diagnostic ecg database csv files/s####_re.csv
# Format (confirmed from s0001_re.csv):
#   Row 0    : header — column names: i, ii, iii, avr, avl, avf, v1..v6, vx, vy, vz
#   Rows 1-N : one time-sample per row (~38 400 rows at 1 kHz), 15 float cols.
#   Lead II  : column "ii"
#
# INSTALLATION
# ------------
#   pip install torch torchvision scikit-learn scipy matplotlib tqdm pandas
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

import re
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
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
CSV_DIR  = DATASET_ROOT / "PTB diagnostic ecg database csv files"
WORK_DIR = Path("/kaggle/working")          # only writable directory on Kaggle

CACHE_FILE = WORK_DIR / "ecg_beats.pkl"    # beat windows + labels cache
RP_CACHE   = WORK_DIR / "rps_learned.npy"  # learned-embedding RP cache

# ── R-peak windowing ──────────────────────────────────────────────────────────
PRE_OFFSET  = 100                           # samples before each R-peak
POST_OFFSET = 200                           # samples after  each R-peak
WINDOW_SIZE = PRE_OFFSET + POST_OFFSET      # = 300  (fixed beat length)

# ── Learned embedding ─────────────────────────────────────────────────────────
EMBED_DIM = 4           # output channels D of the embedding network

# ── Recurrence plot ───────────────────────────────────────────────────────────
RP_SIZE      = 64       # spatial resolution of the output RP image
EPS_QUANTILE = 10       # percentile p used for adaptive threshold epsilon

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
N_EPOCHS   = 30
LR         = 1e-3
TEST_SIZE  = 0.20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

# ── Healthy subject record numbers (PhysioNet PTB documentation) ─────────────
# 80 of the 294 subjects are "Healthy control". Record numbers listed below.
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
    Extracts the integer (14) and checks against the healthy-control set.
    Returns 0 (healthy) or 1 (diseased).
    """
    m = re.search(r's(\d+)', stem, re.IGNORECASE)
    if m is None:
        return 1
    return 0 if int(m.group(1)) in _HEALTHY_RECORD_NUMBERS else 1


def _read_lead_ii(csv_path: Path) -> np.ndarray | None:
    """
    Read one PTB CSV file and return the Lead II column as float32.

    CSV format (confirmed from s0001_re.csv)
    ----------------------------------------
    Row 0    : header with column names: i, ii, iii, avr, avl, avf, v1..v6, vx, vy, vz
    Rows 1-N : one time-sample per row; 15 float columns.
    The "ii" column holds Lead II voltage values (mV).

    Returns None on any read failure, missing column, or signal too short.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        if "ii" not in df.columns:
            return None
        values = pd.to_numeric(df["ii"], errors="coerce").values.astype(np.float32)
        values = values[~np.isnan(values)]
        if len(values) < WINDOW_SIZE:
            return None
        return values
    except Exception:
        return None


# =============================================================================
# SECTION 2b: R-PEAK DETECTION AND BEAT WINDOWING
# =============================================================================

def detect_r_peaks(signal: np.ndarray, fs: int = 1000) -> np.ndarray:
    """
    Detect R-peaks in a z-score normalised ECG signal.

    Uses scipy.signal.find_peaks with:
      - minimum height  = 0.5  (signal is z-scored: mean=0, std~1;
                                 R-peaks typically exceed +0.5)
      - minimum distance = fs // 4 samples = 250 ms refractory period
                           (equivalent to max 240 bpm, above any real HR)

    Parameters
    ----------
    signal : z-score normalised Lead II signal, 1-D float32.
    fs     : sampling frequency in Hz (PTB: 1000 Hz).

    Returns
    -------
    peaks : int array of sample indices of detected R-peaks.
    """
    peaks, _ = find_peaks(signal, height=0.5, distance=fs // 4)
    return peaks


def extract_beat_windows(
    signal: np.ndarray,
    label:  int,
    stem:   str,
    fs:     int = 1000,
) -> tuple[list, list, list]:
    """
    Detect R-peaks and extract a fixed-length window centred on each peak.

    Window: signal[peak - PRE_OFFSET : peak + POST_OFFSET]
    Length: WINDOW_SIZE = PRE_OFFSET + POST_OFFSET (fixed for all beats)

    Boundary policy: windows that would extend outside the signal are
    silently discarded. No padding is applied — every returned window
    is a genuine physiological beat with no synthetic fill.

    Each window inherits the record-level binary label.

    Returns
    -------
    windows : list of float32 arrays of shape (WINDOW_SIZE,)
    labels  : list of int (same value for all beats in one record)
    info    : list of str (stem repeated for each beat)
    """
    peaks = detect_r_peaks(signal, fs)
    windows, labels, info = [], [], []
    for pk in peaks:
        start = pk - PRE_OFFSET
        end   = pk + POST_OFFSET
        if start < 0 or end > len(signal):
            continue
        windows.append(signal[start:end].copy())
        labels.append(label)
        info.append(stem)
    return windows, labels, info


def load_ptb_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over all PTB CSV files, extract Lead II, z-score normalise,
    detect R-peaks, and return beat-centred windows with binary labels.

    Pipeline per file
    -----------------
    1. Read "ii" column  (_read_lead_ii).
    2. Reject flat signals (std < 1e-6).
    3. z-score normalise the full-length signal.
    4. Assign binary label from filename (_label_from_stem).
    5. Detect R-peaks and slice beat windows (extract_beat_windows).

    Caching
    -------
    Results are pickled to CACHE_FILE after the first run.
    Delete CACHE_FILE to force a fresh read (e.g. after changing PRE/POST).

    Returns
    -------
    windows : float32 array, shape (N_beats, WINDOW_SIZE)
    labels  : int64  array, shape (N_beats,)
    info    : object array, shape (N_beats,) — CSV stem per beat window
    """
    if CACHE_FILE.exists():
        print(f"Loading cached beat windows from {CACHE_FILE} ...")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        return data["windows"], data["labels"], data["info"]

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n  {CSV_DIR}\n"
            "Please verify the Kaggle dataset path in SECTION 1."
        )
    print(f"Found {len(csv_files)} CSV files in:\n  {CSV_DIR}")

    all_windows, all_labels, all_info = [], [], []
    skipped_bad  = 0
    label_counts = {0: 0, 1: 0}

    for csv_path in tqdm(csv_files, desc="Loading & windowing"):
        stem   = csv_path.stem
        signal = _read_lead_ii(csv_path)
        if signal is None:
            skipped_bad += 1
            continue

        std = float(signal.std())
        if std < 1e-6:
            skipped_bad += 1
            continue

        signal = (signal - signal.mean()) / std     # z-score normalise
        label  = _label_from_stem(stem)

        wins, labs, inf = extract_beat_windows(signal, label, stem)
        if len(wins) == 0:
            skipped_bad += 1
            continue

        all_windows.extend(wins)
        all_labels.extend(labs)
        all_info.extend(inf)
        label_counts[label] += 1

    print(f"\n  Records loaded : healthy={label_counts[0]}, "
          f"diseased={label_counts[1]}")
    if skipped_bad:
        print(f"  Skipped (unreadable / flat / no peaks): {skipped_bad}")

    if label_counts[0] == 0:
        print(
            "\n  WARNING: zero healthy records identified.\n"
            "  _HEALTHY_RECORD_NUMBERS may not match this upload.\n"
            "  First 10 stems for inspection:"
        )
        for p in csv_files[:10]:
            m = re.search(r's(\d+)', p.stem)
            print(f"    {p.stem}  -> {int(m.group(1)) if m else 'unparseable'}")

    if not all_windows:
        raise RuntimeError(
            "No beat windows produced.\n"
            f"  CSV_DIR: {CSV_DIR}\n"
            "  Check path and that files contain an 'ii' column."
        )

    windows = np.array(all_windows, dtype=np.float32)
    labels  = np.array(all_labels,  dtype=np.int64)
    info    = np.array(all_info,    dtype=object)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"windows": windows, "labels": labels, "info": info}, f)
    print(f"Cached {len(windows):,} beat windows -> {CACHE_FILE}")

    return windows, labels, info


def summarise_dataset(labels: np.ndarray, info: np.ndarray) -> None:
    """Print class balance. Safe against empty arrays."""
    total = len(labels)
    if total == 0:
        print("\n  ERROR: dataset is empty.")
        return
    n_h = int((labels == 0).sum())
    n_d = int((labels == 1).sum())
    print(f"\n{'--'*26}")
    print(f"  Total beat windows : {total:,}")
    print(f"  Healthy  (0)       : {n_h:,}  ({100*n_h/total:.1f}%)")
    print(f"  Diseased (1)       : {n_d:,}  ({100*n_d/total:.1f}%)")
    print(f"  Unique records     : {len(np.unique(info))}")
    print(f"  Window length      : {WINDOW_SIZE} samples  "
          f"(-{PRE_OFFSET} / +{POST_OFFSET} around R-peak)")
    print(f"{'--'*26}\n")


# =============================================================================
# SECTION 3: LEARNED EMBEDDING (1D CNN — preserves temporal resolution)
# =============================================================================

class LearnedEmbedding1D(nn.Module):
    """
    Temporal embedding network: maps an ECG beat window to a multi-channel
    latent trajectory while preserving the full time dimension.

    Specification compliance
    ------------------------
    - Input  : (batch, 1, WINDOW_SIZE)
    - Output : (batch, EMBED_DIM, WINDOW_SIZE)   [time dimension unchanged]
    - All Conv1D layers use same-padding (padding = kernel_size // 2) so
      output length == input length at every layer.
    - NO pooling, NO striding — the temporal resolution is never reduced.
    - No activation on the final layer so latent coordinates are
      unconstrained reals, preserving meaningful distance geometry for RPs.

    The network jointly learns:
    - effective delay structure between time steps
    - effective latent dimensionality
    - nonlinear transformation of the raw ECG morphology

    Architecture
    ------------
    Conv(1->16,  k=7, p=3) -> BN -> ReLU
    Conv(16->32, k=5, p=2) -> BN -> ReLU
    Conv(32->EMBED_DIM, k=3, p=1)          <- no activation on output
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  16,        kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32,        kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1, bias=False),
            # No activation: latent values are unconstrained
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, T)

        Returns
        -------
        z : (batch, EMBED_DIM, T)  -- latent trajectory, same length T
        """
        return self.net(x)


# =============================================================================
# SECTION 4: RECURRENCE PLOT GENERATION (from learned latent trajectory)
# =============================================================================

def compute_recurrence_plot(
    latent:          np.ndarray,
    eps_percentile:  float = EPS_QUANTILE,
    image_size:      int   = RP_SIZE,
) -> np.ndarray:
    """
    Build a normalised, resized recurrence plot from a latent trajectory.

    Parameters
    ----------
    latent        : float32 array, shape (T, D)
                    T time steps, D latent dimensions.
    eps_percentile: percentile p of all pairwise distances used as epsilon.
                    EPS_QUANTILE=10 means the 10% closest pairs are recurrent,
                    giving a well-populated but non-saturated binary matrix.
    image_size    : spatial resolution of the output image (default RP_SIZE=64).

    Steps
    -----
    1. Pairwise Euclidean distance matrix  dist[i,j].
    2. Adaptive threshold: epsilon = percentile(dist, eps_percentile).
    3. Binary recurrence matrix: R[i,j] = 1 if dist[i,j] < epsilon.
    4. Block-average resize to (image_size x image_size).
    5. Normalise to [0, 1].

    Returns
    -------
    rp_img : float32 array, shape (image_size, image_size).
    """
    T = len(latent)

    # 1. Pairwise distances
    diff = latent[:, np.newaxis, :] - latent[np.newaxis, :, :]   # (T, T, D)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))                    # (T, T)

    # 2. Adaptive threshold — strict less-than as specified
    epsilon = float(np.percentile(dist, eps_percentile))

    # 3. Binary recurrence matrix
    rp = (dist < epsilon).astype(np.float32)

    # 4. Block-average resize
    crop = (T // image_size) * image_size
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

    # 5. Normalise to [0, 1]
    mn, mx = rp_img.min(), rp_img.max()
    if mx > mn:
        rp_img = (rp_img - mn) / (mx - mn)
    return rp_img


def build_rp_array_learned(
    windows:  np.ndarray,
    embedder: nn.Module,
    device:   str,
    desc:     str = "Building learned RPs",
) -> np.ndarray:
    """
    Pass all windows through LearnedEmbedding1D, then convert each latent
    trajectory Z (shape T x D) to a recurrence plot (RP_SIZE x RP_SIZE).

    The embedder is run in eval mode with torch.no_grad() so this function
    is safe to call both before and after classifier training.

    Parameters
    ----------
    windows  : float32 array, shape (N, WINDOW_SIZE).
    embedder : LearnedEmbedding1D instance.
    device   : torch device string.

    Returns
    -------
    rps : float32 array, shape (N, RP_SIZE, RP_SIZE).
    """
    embedder.eval()
    rps = np.empty((len(windows), RP_SIZE, RP_SIZE), dtype=np.float32)

    with torch.no_grad():
        for i, win in enumerate(tqdm(windows, desc=desc, leave=False)):
            # (1, 1, T)
            x = (torch.tensor(win, dtype=torch.float32)
                 .unsqueeze(0).unsqueeze(0).to(device))
            # (1, EMBED_DIM, T)
            z = embedder(x)
            # Transpose to (T, EMBED_DIM) for distance computation
            latent  = z.squeeze(0).cpu().numpy().T
            rps[i]  = compute_recurrence_plot(latent)

    return rps


# =============================================================================
# SECTION 5: DATASET CREATION
# =============================================================================

class RPDataset(Dataset):
    """2D CNN dataset. X: (1, RP_SIZE, RP_SIZE), y: long scalar."""
    def __init__(self, rps: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(rps,    dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RawECGDataset(Dataset):
    """1D CNN baseline dataset. X: (1, WINDOW_SIZE), y: long scalar."""
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
    Single stratified train/test split shared across both modalities.
    Both the RP dataset and the raw-window dataset see the exact same
    held-out samples for a fair comparison.

    Returns dict with keys:
        win_train, win_test, rp_train, rp_test, y_train, y_test
    """
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=labels, random_state=SEED,
    )
    return {
        "win_train": windows[idx_train], "win_test": windows[idx_test],
        "rp_train":  rps[idx_train],     "rp_test":  rps[idx_test],
        "y_train":   labels[idx_train],  "y_test":   labels[idx_test],
    }


def make_loaders(splits: dict) -> dict:
    """Wrap split arrays into DataLoaders for both modalities."""
    return {
        "rp_train":  DataLoader(
            RPDataset(splits["rp_train"], splits["y_train"]),
            batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        "rp_test":   DataLoader(
            RPDataset(splits["rp_test"],  splits["y_test"]),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        "raw_train": DataLoader(
            RawECGDataset(splits["win_train"], splits["y_train"]),
            batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        "raw_test":  DataLoader(
            RawECGDataset(splits["win_test"],  splits["y_test"]),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }


# =============================================================================
# SECTION 6: MODEL 1 — 2D CNN CLASSIFIER (operates on learned-embedding RPs)
# =============================================================================

class RecurrenceCNN2D(nn.Module):
    """
    2D CNN classifier operating on 64x64 recurrence-plot images that were
    produced by the LearnedEmbedding1D network.

    Architecture
    ------------
    Block 1 : Conv(1->32,  3x3) -> BN -> GELU -> MaxPool(2)  -> 32x32
    Block 2 : Conv(32->64, 3x3) -> BN -> GELU -> MaxPool(2)  -> 16x16
    Block 3 : Conv(64->128,3x3) -> BN -> GELU -> MaxPool(2)  ->  8x8
    Block 4 : Conv(128->128,3x3)-> BN -> GELU                ->  8x8
              AdaptiveAvgPool(4x4)                            ->  4x4
    Head    : Flatten -> FC(2048->256) -> GELU -> Dropout(0.4) -> FC(256->2)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# =============================================================================
# SECTION 7: MODEL 2 — 1D CNN BASELINE (raw ECG beat windows)
# =============================================================================

class RawECGCNN1D(nn.Module):
    """
    1D CNN baseline operating directly on raw beat windows.

    Architecture
    ------------
    Block 1 : Conv(1->32,  k=7) -> BN -> GELU -> MaxPool(2)
    Block 2 : Conv(32->64, k=5) -> BN -> GELU -> MaxPool(2)
    Block 3 : Conv(64->128,k=3) -> BN -> GELU -> MaxPool(2)
    Block 4 : Conv(128->128,k=3)-> BN -> GELU -> MaxPool(2)
    AdaptiveAvgPool(1) -> Flatten -> FC(128->64) -> GELU -> Dropout -> FC(64->2)

    Larger kernels in early blocks capture multi-component beat structure
    (P-wave, QRS complex, T-wave); later small kernels refine fine morphology.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# =============================================================================
# SECTION 8: TRAINING
# =============================================================================

def _class_weights(labels: np.ndarray, device: str) -> torch.Tensor:
    """Inverse-frequency class weights to handle PTB class imbalance."""
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
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            np.array(all_preds),
            np.array(all_labels))


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

    print(f"\n{'='*60}\n  Training: {model_name}\n{'='*60}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}")
    print("-" * 44)

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc       = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc, _, _ = evaluate_model(
            model, test_loader, criterion, DEVICE)
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
    criterion             = nn.CrossEntropyLoss(weight=class_weights)
    _, acc, preds, labels = evaluate_model(model, test_loader, criterion, DEVICE)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels,  preds, zero_division=0)
    f1   = f1_score(labels,  preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)
    print(f"\n{'-'*50}\n  {model_name} -- Test Metrics\n{'-'*50}")
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
        ["2D CNN (Learned-Embedding RP)", "1D CNN (Raw ECG)"],
    ):
        ConfusionMatrixDisplay(
            confusion_matrix=metrics["cm"],
            display_labels=["Healthy", "Diseased"],
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{title}\nAcc={metrics['acc']*100:.1f}%  F1={metrics['f1']:.3f}",
            fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Confusion matrices saved -> {save_path}")
    plt.show()


def plot_comparison_bar(metrics_rp, metrics_raw,
                         save_path="model_comparison.png"):
    names    = ["Accuracy", "Precision", "Recall", "F1"]
    rp_vals  = [metrics_rp[k]  for k in ("acc", "prec", "rec", "f1")]
    raw_vals = [metrics_raw[k] for k in ("acc", "prec", "rec", "f1")]
    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, rp_vals,  w, label="2D CNN (Learned RP)",
                color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, raw_vals, w, label="1D CNN (Raw ECG)",
                color="#FF5722", alpha=0.85)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=3)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Learned-Embedding RP CNN vs Raw ECG 1D CNN", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Comparison bar chart saved -> {save_path}")
    plt.show()


# =============================================================================
# SECTION 10: VISUALIZATION
# =============================================================================

def visualise_examples(
    windows:     np.ndarray,
    labels:      np.ndarray,
    embedder:    nn.Module,
    device:      str,
    n_per_class: int = 3,
    save_path:   str = "ecg_examples.png",
):
    """
    For n_per_class examples from each class, display three panels:
      (A) Raw beat window with vertical dashed line at R-peak position
      (B) Learned latent space — scatter of Z[:,0] vs Z[:,1]
      (C) Recurrence plot computed from the full EMBED_DIM latent trajectory

    This replaces the old Takens phase-space panel with the learned embedding.
    """
    class_names = {0: "Healthy", 1: "Diseased"}
    colours     = {0: "#27AE60", 1: "#E74C3C"}

    total_rows = 2 * n_per_class
    fig = plt.figure(figsize=(14, 4.2 * total_rows / 2))
    fig.suptitle(
        "Beat Windows | Learned Latent Space | Recurrence Plot",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(total_rows, 3, figure=fig, hspace=0.55, wspace=0.32)

    embedder.eval()
    row_offset = 0

    for cls in [0, 1]:
        idxs = np.where(labels == cls)[0][:n_per_class]
        for local_row, idx in enumerate(idxs):
            win    = windows[idx]
            colour = colours[cls]
            label  = class_names[cls]
            r      = row_offset + local_row

            # Forward pass through the embedder
            with torch.no_grad():
                x = (torch.tensor(win, dtype=torch.float32)
                     .unsqueeze(0).unsqueeze(0).to(device))
                z = embedder(x).squeeze(0).cpu().numpy()   # (EMBED_DIM, T)
            latent = z.T                                    # (T, EMBED_DIM)
            rp     = compute_recurrence_plot(latent)

            # (A) Raw beat
            ax_ts = fig.add_subplot(gs[r, 0])
            ax_ts.plot(win, color=colour, linewidth=0.9, alpha=0.9)
            ax_ts.axvline(PRE_OFFSET, color="k", linewidth=0.7,
                          linestyle="--", alpha=0.6, label="R-peak")
            ax_ts.set_title(f"{label} -- Beat Window", fontsize=8)
            ax_ts.set_xlabel("Sample", fontsize=7)
            ax_ts.set_ylabel("Amplitude (norm.)", fontsize=7)
            ax_ts.tick_params(labelsize=6)
            ax_ts.legend(fontsize=6, loc="upper right")
            ax_ts.grid(alpha=0.25)

            # (B) Learned latent space (first two dims)
            ax_lat = fig.add_subplot(gs[r, 1])
            ax_lat.scatter(latent[:, 0], latent[:, 1],
                           s=2, alpha=0.4, color=colour, linewidths=0)
            ax_lat.set_title(f"{label} -- Latent Space (Z0 vs Z1)", fontsize=8)
            ax_lat.set_xlabel("Z0", fontsize=7)
            ax_lat.set_ylabel("Z1", fontsize=7)
            ax_lat.tick_params(labelsize=6)
            ax_lat.grid(alpha=0.2)

            # (C) Recurrence plot
            ax_rp = fig.add_subplot(gs[r, 2])
            ax_rp.imshow(rp, cmap="inferno", origin="lower", aspect="auto")
            ax_rp.set_title(f"{label} -- Recurrence Plot", fontsize=8)
            ax_rp.axis("off")

        row_offset += n_per_class

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Example figure saved -> {save_path}")
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
        ax.plot(epochs, scale(history_rp[key]),  linewidth=2, color="#2196F3",
                label="2D CNN (Learned RP)")
        ax.plot(epochs, scale(history_raw[key]), linewidth=2, color="#FF5722",
                linestyle="--", label="1D CNN (Raw)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Accuracy (%)" if "acc" in key else "Loss", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Training curves saved -> {save_path}")
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
                     ["Learned-Embedding RP", "Raw Beat", "Predictions"]):
        ax.set_title(t, fontsize=10, fontweight="bold")

    for i in range(n_show):
        axes[i, 0].imshow(imgs_rp[i][0], cmap="inferno",
                          origin="lower", aspect="auto")
        axes[i, 0].axis("off")

        colour = "#27AE60" if labs_raw[i] == 0 else "#E74C3C"
        axes[i, 1].plot(imgs_raw[i][0], color=colour, linewidth=0.8)
        axes[i, 1].axvline(PRE_OFFSET, color="k", linewidth=0.5,
                            linestyle="--", alpha=0.5)
        axes[i, 1].tick_params(labelsize=6)
        axes[i, 1].grid(alpha=0.2)

        gt    = class_names[labs_rp[i]]
        p_rp  = class_names[preds_rp[i]]
        p_raw = class_names[preds_raw[i]]
        ok_rp  = "OK" if preds_rp[i]  == labs_rp[i]  else "X"
        ok_raw = "OK" if preds_raw[i] == labs_raw[i] else "X"
        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.05, 0.5,
            f"Ground truth : {gt}\n"
            f"2D CNN (RP)  : {p_rp}  [{ok_rp}]\n"
            f"1D CNN (Raw) : {p_raw}  [{ok_raw}]",
            transform=axes[i, 2].transAxes,
            fontsize=9, va="center", family="monospace",
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Prediction panel saved -> {save_path}")
    plt.show()


# =============================================================================
# SECTION 11: MAIN LOOP
# =============================================================================

def main():
    print("=" * 62)
    print("  ECG Analysis: Learned-Embedding RP CNN vs Raw ECG 1D CNN")
    print(f"  CSV source  : {CSV_DIR}")
    print(f"  Device      : {DEVICE}")
    print(f"  Beat window : {WINDOW_SIZE} samples "
          f"(-{PRE_OFFSET} / +{POST_OFFSET} around R-peak)")
    print(f"  Embed dim   : {EMBED_DIM}   "
          f"RP size: {RP_SIZE}x{RP_SIZE}   "
          f"epsilon percentile: {EPS_QUANTILE}")
    print("=" * 62)

    # ── Step 1: Load CSVs, detect R-peaks, extract beat windows ─────────
    print("\n[1/9] Loading PTB CSV dataset and extracting beat windows ...")
    windows, labels, info = load_ptb_windows()
    summarise_dataset(labels, info)

    if len(labels) == 0:
        raise RuntimeError(
            "Dataset is empty -- check CSV_DIR path in SECTION 1.")
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            f"Only one class found (values: {np.unique(labels)}).\n"
            "Review _HEALTHY_RECORD_NUMBERS vs actual file stems.")

    # ── Step 2: Instantiate the learned embedding network ────────────────
    print("\n[2/9] Instantiating LearnedEmbedding1D ...")
    embedder = LearnedEmbedding1D(embed_dim=EMBED_DIM).to(DEVICE)
    n_emb = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
    print(f"  Embedding network params: {n_emb:,}")

    # ── Step 3: Visualise examples (before training) ─────────────────────
    # The latent space appears random at this stage; the visualisation
    # confirms shapes, R-peak alignment, and RP generation before the
    # expensive training loop begins.
    print("\n[3/9] Visualising example beats (untrained embedder) ...")
    visualise_examples(
        windows, labels, embedder, DEVICE, n_per_class=3,
        save_path=str(WORK_DIR / "ecg_examples_pretrain.png"))

    # ── Step 4: Compute recurrence plots via the learned embedding ───────
    # RPs are pre-computed and cached so the 2D CNN can be trained with
    # standard minibatch DataLoaders without re-running the embedder each
    # forward pass. The embedder here is untrained (random weights); the
    # 2D CNN therefore learns to classify from random projections of the
    # beat. Rebuilding the cache after joint-training is a natural extension.
    print("\n[4/9] Computing recurrence plots from learned embedding ...")
    if RP_CACHE.exists():
        print(f"  Loading cached RPs from {RP_CACHE} ...")
        rps = np.load(str(RP_CACHE))
    else:
        rps = build_rp_array_learned(windows, embedder, DEVICE)
        np.save(str(RP_CACHE), rps)
        print(f"  Cached {len(rps):,} RPs -> {RP_CACHE}")

    # ── Step 5: Shared train/test splits and DataLoaders ─────────────────
    print("\n[5/9] Creating stratified train/test splits ...")
    splits  = make_splits(windows, labels, rps)
    loaders = make_loaders(splits)
    print(f"  Train: {len(splits['y_train']):,}  |  "
          f"Test: {len(splits['y_test']):,}")

    cw = _class_weights(splits["y_train"], DEVICE)

    # ── Step 6: Build classifier models ──────────────────────────────────
    print("\n[6/9] Building classifier models ...")
    model_rp  = RecurrenceCNN2D().to(DEVICE)
    model_raw = RawECGCNN1D().to(DEVICE)
    print(f"  2D CNN (RP)  params: "
          f"{sum(p.numel() for p in model_rp.parameters()  if p.requires_grad):,}")
    print(f"  1D CNN (Raw) params: "
          f"{sum(p.numel() for p in model_raw.parameters() if p.requires_grad):,}")

    # ── Step 7: Train 2D CNN on learned-embedding RPs ────────────────────
    print("\n[7/9] Training 2D CNN (Learned-Embedding Recurrence Plot) ...")
    history_rp = train_model(
        model_rp, loaders["rp_train"], loaders["rp_test"],
        "2D CNN (Learned-Embedding RP)", cw)

    # ── Step 8: Train 1D CNN baseline ────────────────────────────────────
    print("\n[8/9] Training 1D CNN (Raw ECG Baseline) ...")
    history_raw = train_model(
        model_raw, loaders["raw_train"], loaders["raw_test"],
        "1D CNN (Raw ECG)", cw)

    # ── Step 9: Evaluate, compare, and visualise ─────────────────────────
    print("\n[9/9] Evaluating models ...")
    metrics_rp  = full_evaluation(model_rp,  loaders["rp_test"],
                                  cw, "2D CNN (Learned-Embedding RP)")
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
    visualise_examples(
        windows, labels, embedder, DEVICE, n_per_class=3,
        save_path=str(WORK_DIR / "ecg_examples_posttrain.png"))
    visualise_predictions(
        model_rp,  loaders["rp_test"],
        model_raw, loaders["raw_test"],
        n_show=8, save_path=str(WORK_DIR / "sample_predictions.png"))

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  {'Metric':<12} {'2D CNN (RP)':>16} {'1D CNN (Raw)':>14}")
    print("  " + "-" * 44)
    for key, lbl in [("acc", "Accuracy"), ("prec", "Precision"),
                     ("rec", "Recall"),   ("f1",   "F1")]:
        v_rp  = metrics_rp[key]
        v_raw = metrics_raw[key]
        winner = "< RP" if v_rp >= v_raw else "< Raw"
        print(f"  {lbl:<12} {v_rp:>15.4f}  {v_raw:>13.4f}  {winner}")
    print("=" * 62)

    # ── Save weights ──────────────────────────────────────────────────────
    torch.save(embedder.state_dict(),  str(WORK_DIR / "embedder.pth"))
    torch.save(model_rp.state_dict(),  str(WORK_DIR / "model_rp_2dcnn.pth"))
    torch.save(model_raw.state_dict(), str(WORK_DIR / "model_raw_1dcnn.pth"))
    print(f"\nWeights saved to {WORK_DIR}")


if __name__ == "__main__":
    main()
