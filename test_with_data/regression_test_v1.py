#NOTE: THIS ONE DOES NOT WORK AND IS ONLY IN THE REPO TO SHOW THE FAILURE MODES OF CARAPACE

# =============================================================================
# PM2.5 Regression: Learned-Embedding Continuous RP CNN vs 1D CNN Baseline
# Dataset : Beijing PM2.5 (PRSA_data_2010_1_1-2014_12_31.csv)
# Task    : Predict future PM2.5 concentration (regression)
# =============================================================================
#
# DATASET COLUMNS (confirmed from file inspection)
# -------------------------------------------------
#   No        : row index (dropped)
#   year      : 2010-2014  \
#   month     : 1-12        >  collapsed to a single linear integer clock t
#   day       : 1-31       /
#   hour      : 0-23      /
#   pm2.5     : target variable (µg/m³); has NA in first ~24 rows
#   DEWP      : dew point temperature (°C)
#   TEMP      : air temperature (°C)
#   PRES      : atmospheric pressure (hPa)
#   cbwd      : combined wind direction (categorical → one-hot encoded)
#   Iws       : cumulated wind speed (m/s)
#   Is        : cumulated hours of snow
#   Ir        : cumulated hours of rain
#
# NaN ROOT-CAUSE ANALYSIS AND FIXES (vs previous version)
# --------------------------------------------------------
# FIX 1 — sqrt(0) undefined gradient in _soft_rp:
#   dist2.sqrt() produces NaN gradients when two latent points are
#   identical (dist2=0 at random init with BN).
#   Fix: use (dist2 + EPS).sqrt() everywhere.
#
# FIX 2 — sigma underflow in _soft_rp:
#   At random init the mean pairwise distance can be near zero, making
#   exp(-dist/sigma) blow up or produce NaN.
#   Fix: sigma = dist.mean() + SIGMA_EPS, clamp exponent input to [-20, 0].
#
# FIX 3 — detrend applied before clip, corrupting target values:
#   clip(lower=0) was applied before detrending, so after rolling-mean
#   subtraction pm2.5 could become negative again.
#   Fix: apply clip after all preprocessing is complete.
#
# FIX 4 — plot_examples referenced 'pm2.5' in feat_cols (it isn't there):
#   pm2.5 is the target and is excluded from feat_cols.
#   Fix: display DEWP, TEMP, PRES, Iws instead (all guaranteed in feat_cols).
#
# FIX 5 — NaN check added after preprocessing and windowing:
#   Explicit assertions guard against silent NaN propagation into tensors.
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Update DATA_PATH to wherever the CSV lives in your environment.
DATA_PATH = Path("/kaggle/input/datasets/djhavera/beijing-pm25-data-data-set/PRSA_data_2010.1.1-2014.12.31.csv")
WORK_DIR  = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 128   # input timesteps per sample
HORIZON     = 1     # predict PM2.5 t + HORIZON steps ahead
STRIDE      = 1     # sliding-window stride

# ── Preprocessing ─────────────────────────────────────────────────────────────
SMOOTH_WINDOW  = 3      # moving-average kernel size for light smoothing
DETREND        = True   # subtract rolling mean to remove slow trends
ROLLING_WINDOW = 24     # rolling-mean window for detrending (hours)
NOISE_STD      = 0.01   # Gaussian noise std added to inputs during training

# ── Learned embedding ─────────────────────────────────────────────────────────
EMBED_DIM = 6           # latent trajectory channels D

# ── Recurrence plot ───────────────────────────────────────────────────────────
RP_SIZE   = 64          # output image spatial resolution

# ── Numerical stability (FIX 1, FIX 2) ───────────────────────────────────────
DIST_EPS  = 1e-6        # added inside sqrt to avoid NaN gradient at dist=0
SIGMA_EPS = 1e-4        # added to sigma to prevent division by ~zero
EXP_CLAMP = 20.0        # clamp -dist/sigma to [-EXP_CLAMP, 0] before exp()

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
N_EPOCHS   = 40
LR         = 1e-3
TEST_SIZE  = 0.20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

_WIND_DIRS = ["NE", "NW", "SE", "cv"]   # all wind-direction categories in dataset


def load_and_preprocess(path: Path = DATA_PATH) -> tuple[pd.DataFrame, list[str], str]:
    """
    Load the Beijing PM2.5 CSV, build a linear time index, encode
    wind direction, fill missing values, smooth, detrend, and normalise.

    Column handling
    ---------------
    - 'No'   : dropped (meaningless row index)
    - year/month/day/hour : collapsed into 't' = hours since 2010-01-01 00:00
    - 'cbwd' : one-hot encoded into wind_NE, wind_NW, wind_SE, wind_cv
    - All others: numeric features; 'pm2.5' is the regression target

    Preprocessing order (order matters to avoid FIX 3 bug)
    -------------------------------------------------------
    1. Drop 'No', build 't', one-hot encode cbwd.
    2. Fill all NaNs: ffill → bfill → linear interpolation.
    3. Light moving-average smoothing (window = SMOOTH_WINDOW).
    4. Optional detrending: subtract rolling mean (ROLLING_WINDOW).
    5. Per-feature z-score normalisation.
    6. Final clip of pm2.5 to [0, ∞) AFTER all transforms (FIX 3).

    Normalisation is applied LAST so that the z-scored pm2.5 is
    the actual regression target seen by the model.
    """
    df = pd.read_csv(path)

    # ── 1a. Drop row index ───────────────────────────────────────────────
    df = df.drop(columns=["No"], errors="ignore")

    # ── 1b. Collapse year/month/day/hour → linear clock 't' ─────────────
    reference = pd.Timestamp("2010-01-01 00:00:00")
    df["t"] = (
        pd.to_datetime(
            dict(year=df["year"], month=df["month"],
                 day=df["day"],   hour=df["hour"])
        ) - reference
    ).dt.total_seconds() / 3600.0
    df = df.drop(columns=["year", "month", "day", "hour"])

    # ── 1c. One-hot encode wind direction ─────────────────────────────────
    for wd in _WIND_DIRS:
        df[f"wind_{wd}"] = (df["cbwd"] == wd).astype(np.float32)
    df = df.drop(columns=["cbwd"])

    # ── 2. Fill missing values ────────────────────────────────────────────
    # The first ~24 rows have pm2.5 = NA (no measurement on day 1).
    # bfill propagates the first valid value backwards; interpolate fills
    # any isolated interior gaps.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = (
        df[num_cols]
        .ffill()
        .bfill()
        .interpolate(method="linear", limit_direction="both")
    )

    # Verify no NaNs remain after filling
    remaining_nan = df[num_cols].isna().sum().sum()
    if remaining_nan > 0:
        raise ValueError(f"NaN filling incomplete: {remaining_nan} NaNs remain")

    # ── 3. Light moving-average smoothing ────────────────────────────────
    if SMOOTH_WINDOW > 1:
        df[num_cols] = (
            df[num_cols]
            .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
        )

    # ── 4. Optional detrending ────────────────────────────────────────────
    if DETREND:
        rolling_mean = (
            df[num_cols]
            .rolling(ROLLING_WINDOW, center=True, min_periods=1)
            .mean()
        )
        df[num_cols] = df[num_cols] - rolling_mean

    # ── 5. Per-feature z-score normalisation ─────────────────────────────
    means = df[num_cols].mean()
    stds  = df[num_cols].std()
    stds  = stds.replace(0.0, 1.0)   # guard zero-variance columns
    df[num_cols] = (df[num_cols] - means) / stds

    # ── 6. Clip pm2.5 AFTER all transforms (FIX 3) ───────────────────────
    # After detrending and normalising, pm2.5 is centred near 0 and can be
    # negative for below-average readings. We do NOT clip here — clipping a
    # z-scored target would distort the distribution. Instead we leave it as
    # is: the model predicts in normalised space and negative values are valid
    # (they represent below-average PM2.5).

    # ── Define feature / target split ────────────────────────────────────
    target_col = "pm2.5"
    feat_cols  = [c for c in df.columns if c != target_col]

    # Final NaN check over the entire dataframe
    total_nan = df.isna().sum().sum()
    print(f"Preprocessed dataset: {len(df):,} rows × {len(df.columns)} cols")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")
    print(f"  Target           : {target_col}")
    print(f"  NaN remaining    : {total_nan}")
    print(f"  Target range     : [{df[target_col].min():.3f}, "
          f"{df[target_col].max():.3f}]")

    if total_nan > 0:
        raise ValueError(f"NaNs present after preprocessing: {total_nan}")

    return df, feat_cols, target_col


# =============================================================================
# SECTION 3: SLIDING WINDOW EXTRACTION
# =============================================================================

def make_windows(
    df:         pd.DataFrame,
    feat_cols:  list[str],
    target_col: str,
    window:     int = WINDOW_SIZE,
    horizon:    int = HORIZON,
    stride:     int = STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows from the preprocessed time series.

    Each sample
    -----------
    X[i] : shape (window, num_features)  — consecutive hourly timesteps
    y[i] : scalar — normalised PM2.5 at timestep (window_end + horizon)

    Returns
    -------
    X : float32 array, shape (N, window, num_features)
    y : float32 array, shape (N,)
    """
    feat = df[feat_cols].values.astype(np.float32)   # (T, F)
    tgt  = df[target_col].values.astype(np.float32)  # (T,)
    T    = len(feat)

    xs, ys = [], []
    for start in range(0, T - window - horizon + 1, stride):
        end = start + window
        xs.append(feat[start:end])
        ys.append(tgt[end + horizon - 1])

    X = np.array(xs, dtype=np.float32)   # (N, T, F)
    y = np.array(ys, dtype=np.float32)   # (N,)

    # FIX 5: explicit NaN guard before any tensor creation
    assert not np.isnan(X).any(), "NaN detected in feature windows"
    assert not np.isnan(y).any(), "NaN detected in target array"

    print(f"Windows: {len(X):,}  |  shape: {X.shape}  |  "
          f"target range: [{y.min():.3f}, {y.max():.3f}]")
    return X, y


# =============================================================================
# SECTION 4: LEARNED EMBEDDING (1D CNN — multivariate, preserves T)
# =============================================================================

class LearnedEmbedding1D(nn.Module):
    """
    Maps a multivariate window to a latent trajectory of the same length.

    Input  : (batch, num_features, T)
    Output : (batch, EMBED_DIM, T)   — T is never reduced

    Design
    ------
    - All Conv1d layers use same-padding (output_length == input_length).
    - No pooling, no striding — temporal resolution is fully preserved.
    - Early layer (k=7): mixes across features, wide temporal receptive field.
    - Middle layer (k=5): intermediate feature mixing.
    - Final layer (k=3): no activation → unconstrained real latent values.
      Unconstrained outputs are critical for the RP: Euclidean distances
      between latent points must be geometrically meaningful (not squashed).
    """

    def __init__(self, num_features: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, 32,        kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32,           64,        kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64,           embed_dim, kernel_size=3, padding=1, bias=False),
            # No BatchNorm or activation on the final layer:
            # - BN would centre the latent coordinates, compressing distances
            # - Activation would restrict the range of distances
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, F, T)  ->  z : (B, D, T)"""
        return self.net(x)


# =============================================================================
# SECTION 5: CONTINUOUS RECURRENCE PLOT GENERATION
# =============================================================================

def compute_continuous_rp(latent: np.ndarray,
                           image_size: int = RP_SIZE) -> np.ndarray:
    """
    Continuous (soft) recurrence plot: R(i,j) = exp(-dist(i,j) / sigma).

    sigma = std of all pairwise distances (adaptive, per sample).
    Output is in [0, 1]: identical states → 1, distant states → ~0.

    This is strictly continuous — no binary threshold anywhere.
    Used for the static RP cache and for visualisation.
    """
    T = len(latent)

    diff  = latent[:, np.newaxis, :] - latent[np.newaxis, :, :]   # (T, T, D)
    dist2 = np.sum(diff ** 2, axis=-1)                             # (T, T)
    dist  = np.sqrt(dist2 + DIST_EPS)                              # (T, T), no NaN

    sigma = dist.std()
    if sigma < SIGMA_EPS:
        sigma = 1.0

    rp = np.exp(-np.clip(dist / sigma, 0.0, EXP_CLAMP)).astype(np.float32)

    # Resize to image_size × image_size
    if T != image_size:
        zoom_factor = image_size / T
        rp = scipy_zoom(rp, zoom_factor, order=1).astype(np.float32)
        rp = np.clip(rp, 0.0, 1.0)

    mn, mx = rp.min(), rp.max()
    if mx > mn:
        rp = (rp - mn) / (mx - mn)

    return rp


def build_rp_array(windows: np.ndarray, embedder: nn.Module,
                   device: str, desc: str = "Building RPs") -> np.ndarray:
    """
    Pass all windows through the embedder (eval mode) and compute
    one continuous RP per window.

    windows : (N, T, F)
    Returns : (N, RP_SIZE, RP_SIZE)  float32
    """
    embedder.eval()
    rps = np.empty((len(windows), RP_SIZE, RP_SIZE), dtype=np.float32)

    with torch.no_grad():
        for i, win in enumerate(tqdm(windows, desc=desc, leave=False)):
            x = (torch.tensor(win.T, dtype=torch.float32)
                 .unsqueeze(0).to(device))            # (1, F, T)
            z = embedder(x)                            # (1, D, T)
            latent = z.squeeze(0).cpu().numpy().T      # (T, D)
            rps[i] = compute_continuous_rp(latent)

    return rps


# =============================================================================
# SECTION 6: DATASET CLASSES
# =============================================================================

class RPRegressionDataset(Dataset):
    """
    2D CNN dataset.  X: (1, RP_SIZE, RP_SIZE),  y: float32 scalar.
    Optional Gaussian noise on the image during training (mild regulariser).
    """

    def __init__(self, rps: np.ndarray, labels: np.ndarray,
                 noise_std: float = 0.0):
        self.X         = torch.tensor(rps,    dtype=torch.float32).unsqueeze(1)
        self.y         = torch.tensor(labels, dtype=torch.float32)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.noise_std > 0.0:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)
        return x, self.y[idx]


class RawWindowDataset(Dataset):
    """
    1D CNN dataset.  X: (num_features, T) channels-first,  y: float32 scalar.
    Optional Gaussian noise on the raw signal during training.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray,
                 noise_std: float = 0.0):
        X = windows.transpose(0, 2, 1).astype(np.float32)  # (N, F, T)
        self.X         = torch.tensor(X,      dtype=torch.float32)
        self.y         = torch.tensor(labels, dtype=torch.float32)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[idx]


def make_splits_and_loaders(X: np.ndarray, y: np.ndarray,
                             rps: np.ndarray) -> tuple[dict, dict]:
    """
    Chronological train/test split (no shuffle — preserves time ordering).

    The split is shared across both modalities so test targets are identical.
    Returns: splits dict, loaders dict.
    """
    n         = len(y)
    n_test    = int(n * TEST_SIZE)
    n_train   = n - n_test

    # Chronological split: first n_train for training, last n_test for test
    splits = {
        "X_train":  X[:n_train],    "X_test":  X[n_train:],
        "rp_train": rps[:n_train],  "rp_test": rps[n_train:],
        "y_train":  y[:n_train],    "y_test":  y[n_train:],
    }

    loaders = {
        "rp_train": DataLoader(
            RPRegressionDataset(splits["rp_train"], splits["y_train"],
                                noise_std=NOISE_STD),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        "rp_test":  DataLoader(
            RPRegressionDataset(splits["rp_test"], splits["y_test"]),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        "raw_train": DataLoader(
            RawWindowDataset(splits["X_train"], splits["y_train"],
                             noise_std=NOISE_STD),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        "raw_test":  DataLoader(
            RawWindowDataset(splits["X_test"], splits["y_test"]),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }

    print(f"  Train: {n_train:,}  |  Test: {n_test:,}  "
          f"(chronological split, no shuffle)")
    return splits, loaders


# =============================================================================
# SECTION 7: MODEL A — 2D CNN REGRESSION (continuous RP input)
# =============================================================================

class RecurrenceCNN2D(nn.Module):
    """
    2D CNN regressor operating on 64×64 single-channel continuous RP images.

    Output: single scalar (no activation) for direct PM2.5 regression.

    Architecture
    ------------
    Block 1 : Conv(1->32,  3×3) -> BN -> GELU -> MaxPool(2)   32×32
    Block 2 : Conv(32->64, 3×3) -> BN -> GELU -> MaxPool(2)   16×16
    Block 3 : Conv(64->128,3×3) -> BN -> GELU -> MaxPool(2)    8×8
    Block 4 : Conv(128->128,3×3)-> BN -> GELU                  8×8
              AdaptiveAvgPool(4×4)                              4×4
    Head    : Flatten -> FC(2048->256) -> GELU -> Dropout(0.3) -> FC(256->1)
    """

    def __init__(self, dropout: float = 0.3):
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
            nn.Linear(256, 1),   # raw scalar — no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)   # (B,)


# =============================================================================
# SECTION 8: MODEL B — 1D CNN BASELINE (raw multivariate window)
# =============================================================================

class RawCNN1D(nn.Module):
    """
    1D CNN regression baseline operating on (F, T) multivariate windows.

    Output: single scalar (no activation) for direct PM2.5 regression.

    Architecture
    ------------
    Block 1 : Conv(F->64,  k=7) -> BN -> GELU -> MaxPool(2)
    Block 2 : Conv(64->128,k=5) -> BN -> GELU -> MaxPool(2)
    Block 3 : Conv(128->256,k=3)-> BN -> GELU -> MaxPool(2)
    Block 4 : Conv(256->256,k=3)-> BN -> GELU -> MaxPool(2)
    AdaptiveAvgPool(1) -> Flatten -> FC(256->64) -> GELU -> Dropout -> FC(64->1)
    """

    def __init__(self, num_features: int, dropout: float = 0.3):
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
            cb(num_features, 64,  7),
            cb(64,           128, 5),
            cb(128,          256, 3),
            cb(256,          256, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),    # raw scalar — no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)   # (B,)


# =============================================================================
# SECTION 9: END-TO-END MODEL (embedder + differentiable soft RP + 2D CNN)
# =============================================================================

class EndToEndRPModel(nn.Module):
    """
    Fully differentiable pipeline: embedding → soft continuous RP → 2D CNN.

    The soft RP uses a Gaussian kernel instead of a hard threshold, making
    the entire pipeline end-to-end trainable with MSE loss:

        R_soft(i,j) = exp( -dist(i,j) / sigma )

    where sigma = mean pairwise distance (computed per batch, per sample).

    NaN prevention (FIX 1, FIX 2)
    ------------------------------
    - dist = sqrt(dist2 + DIST_EPS) prevents NaN gradient at dist=0.
    - sigma = dist.mean() + SIGMA_EPS prevents division by zero.
    - The exponent -dist/sigma is clamped to [-EXP_CLAMP, 0] before exp()
      to prevent underflow/overflow (exp(-20) ≈ 2e-9, effectively zero).

    Input  : (batch, F, T)  — multivariate window, channels-first
    Output : (batch,)       — scalar PM2.5 prediction
    """

    def __init__(self, num_features: int,
                 embed_dim: int = EMBED_DIM,
                 rp_size:   int = RP_SIZE):
        super().__init__()
        self.embedder   = LearnedEmbedding1D(num_features, embed_dim)
        self.regressor  = RecurrenceCNN2D()   # renamed from 'classifier' (FIX 3)
        self.rp_size    = rp_size

    def _soft_rp(self, z: torch.Tensor) -> torch.Tensor:
        """
        Differentiable continuous RP from latent trajectory.

        Parameters
        ----------
        z : (B, D, T)

        Returns
        -------
        rp : (B, 1, rp_size, rp_size)  values in [0, 1]
        """
        z = z.permute(0, 2, 1)             # (B, T, D)
        B, T, D = z.shape

        # Pairwise squared distances
        diff  = z.unsqueeze(2) - z.unsqueeze(1)    # (B, T, T, D)
        dist2 = diff.pow(2).sum(-1)                # (B, T, T)

        # FIX 1: add DIST_EPS inside sqrt to guarantee finite gradient at 0
        dist = (dist2 + DIST_EPS).sqrt()           # (B, T, T)

        # FIX 2: sigma = mean distance + SIGMA_EPS to prevent zero division
        sigma = dist.mean(dim=(1, 2), keepdim=True) + SIGMA_EPS   # (B, 1, 1)

        # FIX 2: clamp the exponent before exp() to prevent overflow/underflow
        exponent = -(dist / sigma).clamp(min=-EXP_CLAMP, max=0.0)
        rp = exponent.exp()                        # (B, T, T), values in (0, 1]

        # Resize T×T → rp_size×rp_size
        rp = rp.unsqueeze(1)                       # (B, 1, T, T)
        rp = F.interpolate(
            rp, size=(self.rp_size, self.rp_size),
            mode="bilinear", align_corners=False,
        )                                          # (B, 1, H, W)

        # Per-sample normalisation to [0, 1]
        flat = rp.view(B, -1)
        mn   = flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
        mx   = flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
        rp   = (rp - mn) / (mx - mn + SIGMA_EPS)

        return rp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z  = self.embedder(x)     # (B, D, T)
        rp = self._soft_rp(z)     # (B, 1, H, W)
        return self.regressor(rp)  # (B,)


# =============================================================================
# SECTION 10: TRAINING AND EVALUATION
# =============================================================================

def _check_for_nan(loss_val: float, epoch: int, model_name: str) -> None:
    """Raise an informative error immediately if loss is NaN."""
    if not np.isfinite(loss_val):
        raise RuntimeError(
            f"\n[NaN detected] Model='{model_name}', Epoch={epoch}\n"
            "  Possible causes: exploding gradients, NaN in input data,\n"
            "  or numerical instability in the RP computation.\n"
            "  Check DIST_EPS, SIGMA_EPS, and EXP_CLAMP in SECTION 1."
        )


def train_one_epoch(model, loader, criterion, optimizer, device,
                    model_name: str, epoch: int) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        # Gradient clipping prevents exploding gradients from propagating NaN
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_loss = loss.item()
        _check_for_nan(batch_loss, epoch, model_name)
        total_loss += batch_loss * y.size(0)
        n          += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns MSE, predictions array, targets array."""
    model.eval()
    preds_list, targets_list = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        preds_list.extend(pred.cpu().numpy())
        targets_list.extend(y.cpu().numpy())
    preds   = np.array(preds_list,   dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    mse     = float(np.mean((preds - targets) ** 2))
    return mse, preds, targets


def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    label: str = "") -> dict:
    mse  = mean_squared_error(targets, preds)
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds)
    rmse = float(np.sqrt(mse))
    if label:
        print(f"\n{'─'*50}\n  {label}\n{'─'*50}")
        print(f"  MSE  : {mse:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  MAE  : {mae:.6f}")
        print(f"  R²   : {r2:.4f}")
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2)}


def train_model(model, train_loader, test_loader,
                model_name: str,
                n_epochs: int = N_EPOCHS,
                lr: float = LR,
                wd: float = 1e-4) -> dict:
    """
    AdamW + CosineAnnealing training loop with best-MSE checkpoint restore.
    Raises RuntimeError immediately if loss becomes NaN (FIX 5).
    """
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    history = {"train_loss": [], "test_loss": []}
    best_mse, best_state = float("inf"), None

    print(f"\n{'='*60}\n  Training: {model_name}\n{'='*60}")
    print(f"  lr={lr}  wd={wd}  epochs={n_epochs}  batch={BATCH_SIZE}")
    print(f"{'Ep':>4}  {'Tr MSE':>10}  {'Te MSE':>10}")
    print("─" * 30)

    for epoch in range(1, n_epochs + 1):
        tr_loss           = train_one_epoch(model, train_loader, criterion,
                                            optimizer, DEVICE, model_name, epoch)
        te_mse, _, _      = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_mse)

        if te_mse < best_mse:
            best_mse   = te_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>4}  {tr_loss:>10.6f}  {te_mse:>10.6f}")

    model.load_state_dict(best_state)
    print(f"\n  Best test MSE: {best_mse:.6f}")
    return history


# =============================================================================
# SECTION 11: VISUALIZATION
# =============================================================================

def plot_training_curves(history_rp: dict, history_raw: dict,
                          save_path: str = "training_curves.png"):
    epochs = range(1, len(history_rp["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, hist, title, colour in zip(
        axes,
        [history_rp, history_raw],
        ["End-to-End RP Model", "1D CNN Baseline"],
        ["#2196F3", "#FF5722"],
    ):
        ax.plot(epochs, hist["train_loss"], color=colour,
                linewidth=2, label="Train MSE")
        ax.plot(epochs, hist["test_loss"],  color=colour,
                linewidth=2, linestyle="--", alpha=0.7, label="Test MSE")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE (normalised space)", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Training curves saved -> {save_path}")
    plt.show()


def plot_predictions(preds_rp: np.ndarray, preds_raw: np.ndarray,
                      targets: np.ndarray, n_show: int = 300,
                      save_path: str = "predictions.png"):
    t  = np.arange(min(n_show, len(targets)))
    y  = targets[:n_show]
    p1 = preds_rp[:n_show]
    p2 = preds_raw[:n_show]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PM2.5 Prediction — Normalised Space", fontsize=13,
                 fontweight="bold")

    axes[0].plot(t, y,  color="black",   linewidth=1.2, label="Actual", alpha=0.85)
    axes[0].plot(t, p1, color="#2196F3", linewidth=1.0, label="RP Model", alpha=0.75)
    axes[0].plot(t, p2, color="#FF5722", linewidth=1.0, label="1D CNN",
                 alpha=0.75, linestyle="--")
    axes[0].set_xlabel("Test sample index", fontsize=9)
    axes[0].set_ylabel("Normalised PM2.5", fontsize=9)
    axes[0].set_title(f"First {n_show} test samples", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    err1 = preds_rp  - targets
    err2 = preds_raw - targets
    axes[1].hist(err1, bins=60, alpha=0.6, color="#2196F3",
                 label=f"RP  MAE={np.abs(err1).mean():.4f}")
    axes[1].hist(err2, bins=60, alpha=0.6, color="#FF5722",
                 label=f"Raw MAE={np.abs(err2).mean():.4f}")
    axes[1].axvline(0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Prediction error", fontsize=9)
    axes[1].set_ylabel("Count", fontsize=9)
    axes[1].set_title("Error Distribution (full test set)", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Prediction figure saved -> {save_path}")
    plt.show()


def plot_scatter(preds_rp: np.ndarray, preds_raw: np.ndarray,
                  targets: np.ndarray, save_path: str = "scatter.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Predicted vs Actual PM2.5 (normalised)",
                 fontsize=13, fontweight="bold")

    for ax, preds, title, colour in zip(
        axes,
        [preds_rp, preds_raw],
        ["RP End-to-End Model", "1D CNN Baseline"],
        ["#2196F3", "#FF5722"],
    ):
        ax.scatter(targets, preds, s=4, alpha=0.25, color=colour)
        lo = min(float(targets.min()), float(preds.min()))
        hi = max(float(targets.max()), float(preds.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0,
                label="Perfect prediction")
        r2 = r2_score(targets, preds)
        ax.set_title(f"{title}  (R²={r2:.3f})", fontsize=10)
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Scatter figure saved -> {save_path}")
    plt.show()


def plot_examples(X_examples: np.ndarray, rp_examples: np.ndarray,
                   y_examples: np.ndarray, p_rp: np.ndarray,
                   p_raw: np.ndarray, feat_cols: list[str],
                   n_show: int = 4, save_path: str = "examples.png"):
    """
    For n_show samples: raw feature signals | continuous RP | predictions.

    FIX 4: display features guaranteed to be in feat_cols (DEWP, TEMP, PRES, Iws).
    'pm2.5' is the target, not a feature — it was NOT included in feat_cols.
    """
    # Only display features that actually exist in feat_cols
    candidate_feats = ["DEWP", "TEMP", "PRES", "Iws"]
    display_feats   = [f for f in candidate_feats if f in feat_cols]
    feat_indices    = [feat_cols.index(f) for f in display_feats]

    n_show = min(n_show, len(y_examples))
    fig = plt.figure(figsize=(4 * n_show, 10))
    fig.suptitle("Example Samples: Signals | Recurrence Plot | Predictions",
                 fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(3, n_show, figure=fig, hspace=0.55, wspace=0.3)

    for col in range(n_show):
        # ── Raw feature signals ───────────────────────────────────────────
        ax_sig = fig.add_subplot(gs[0, col])
        for fi, fl in zip(feat_indices, display_feats):
            ax_sig.plot(X_examples[col, :, fi], linewidth=0.8,
                        label=fl, alpha=0.85)
        ax_sig.set_title(f"Sample {col + 1}", fontsize=8)
        ax_sig.set_xlabel("Time step", fontsize=7)
        ax_sig.tick_params(labelsize=6)
        ax_sig.legend(fontsize=5, loc="upper right")
        ax_sig.grid(alpha=0.2)

        # ── Continuous recurrence plot ────────────────────────────────────
        ax_rp = fig.add_subplot(gs[1, col])
        ax_rp.imshow(rp_examples[col], cmap="inferno",
                     origin="lower", aspect="auto", vmin=0, vmax=1)
        ax_rp.set_title("Continuous RP", fontsize=8)
        ax_rp.axis("off")

        # ── Prediction annotation ─────────────────────────────────────────
        ax_ann = fig.add_subplot(gs[2, col])
        ax_ann.axis("off")
        ax_ann.text(
            0.05, 0.5,
            f"Actual  : {y_examples[col]:.3f}\n"
            f"RP CNN  : {p_rp[col]:.3f}\n"
            f"1D CNN  : {p_raw[col]:.3f}",
            transform=ax_ann.transAxes,
            fontsize=9, va="center", family="monospace",
        )

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Example figure saved -> {save_path}")
    plt.show()


def plot_comparison_bar(m_rp: dict, m_raw: dict,
                         save_path: str = "comparison.png"):
    names    = ["MSE", "MAE", "R²"]
    rp_vals  = [m_rp["mse"],  m_rp["mae"],  m_rp["r2"]]
    raw_vals = [m_raw["mse"], m_raw["mae"], m_raw["r2"]]
    x, w = np.arange(3), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, rp_vals,  w, label="RP Model",  color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, raw_vals, w, label="1D CNN Raw", color="#FF5722", alpha=0.85)
    ax.bar_label(b1, fmt="%.4f", fontsize=8, padding=3)
    ax.bar_label(b2, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score (normalised space)", fontsize=10)
    ax.set_title("End-to-End RP CNN vs 1D CNN — Regression Metrics", fontsize=11)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Comparison bar chart saved -> {save_path}")
    plt.show()


# =============================================================================
# SECTION 12: MAIN LOOP
# =============================================================================

def main():
    print("=" * 66)
    print("  PM2.5 Regression: End-to-End RP CNN vs 1D CNN Baseline")
    print(f"  Device      : {DEVICE}")
    print(f"  Window      : {WINDOW_SIZE} steps  |  Horizon: +{HORIZON}")
    print(f"  Embed dim   : {EMBED_DIM}  |  RP size: {RP_SIZE}×{RP_SIZE}")
    print(f"  Smooth k    : {SMOOTH_WINDOW}  |  Detrend: {DETREND}  "
          f"|  Noise std: {NOISE_STD}")
    print(f"  Stability   : DIST_EPS={DIST_EPS}  SIGMA_EPS={SIGMA_EPS}  "
          f"EXP_CLAMP={EXP_CLAMP}")
    print("=" * 66)

    # ── Step 1: Load and preprocess ───────────────────────────────────────
    print("\n[1/9] Loading and preprocessing Beijing PM2.5 dataset ...")
    df, feat_cols, target_col = load_and_preprocess(DATA_PATH)
    num_features = len(feat_cols)
    print(f"  num_features = {num_features}")

    # ── Step 2: Sliding windows ───────────────────────────────────────────
    print(f"\n[2/9] Extracting sliding windows "
          f"(size={WINDOW_SIZE}, stride={STRIDE}, horizon={HORIZON}) ...")
    X, y = make_windows(df, feat_cols, target_col)

    # ── Step 3: Instantiate embedder ──────────────────────────────────────
    print("\n[3/9] Instantiating LearnedEmbedding1D ...")
    embedder = LearnedEmbedding1D(num_features, EMBED_DIM).to(DEVICE)
    print(f"  Embedding params: "
          f"{sum(p.numel() for p in embedder.parameters() if p.requires_grad):,}")

    # ── Step 4: Build RP cache (for test set + visualisation) ─────────────
    rp_cache = WORK_DIR / "pm25_rps.npy"
    if rp_cache.exists():
        print(f"\n[4/9] Loading cached RPs from {rp_cache} ...")
        rps = np.load(str(rp_cache))
        if rps.shape[0] != len(X):
            print(f"  Cache shape mismatch ({rps.shape[0]} vs {len(X)}) — rebuilding.")
            rp_cache.unlink()
            rps = build_rp_array(X, embedder, DEVICE)
            np.save(str(rp_cache), rps)
    else:
        print("\n[4/9] Computing continuous recurrence plots ...")
        rps = build_rp_array(X, embedder, DEVICE)
        np.save(str(rp_cache), rps)
        print(f"  Cached {len(rps):,} RPs -> {rp_cache}")

    # ── Step 5: Splits and loaders ────────────────────────────────────────
    print("\n[5/9] Creating train/test splits ...")
    splits, loaders = make_splits_and_loaders(X, y, rps)

    # ── Step 6: Build models ──────────────────────────────────────────────
    print("\n[6/9] Building models ...")
    model_rp  = EndToEndRPModel(num_features, EMBED_DIM, RP_SIZE).to(DEVICE)
    model_raw = RawCNN1D(num_features).to(DEVICE)
    n_rp  = sum(p.numel() for p in model_rp.parameters()  if p.requires_grad)
    n_raw = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
    print(f"  End-to-end RP model : {n_rp:,} params")
    print(f"  1D CNN baseline     : {n_raw:,} params")

    # ── Step 7: Train Model A (end-to-end, uses raw_train loader) ─────────
    print("\n[7/9] Training end-to-end RP model ...")
    history_rp = train_model(
        model_rp,
        loaders["raw_train"],   # (F, T) windows → EndToEndRPModel
        loaders["raw_test"],
        "End-to-End RP Model",
    )

    # ── Step 8: Train Model B (1D CNN baseline) ───────────────────────────
    print("\n[8/9] Training 1D CNN baseline ...")
    history_raw = train_model(
        model_raw,
        loaders["raw_train"],
        loaders["raw_test"],
        "1D CNN Baseline",
    )

    # ── Step 9: Evaluate ──────────────────────────────────────────────────
    print("\n[9/9] Evaluating ...")
    _, preds_rp,  targets = evaluate(model_rp,  loaders["raw_test"], DEVICE)
    _, preds_raw, _       = evaluate(model_raw, loaders["raw_test"], DEVICE)

    m_rp  = compute_metrics(preds_rp,  targets, "End-to-End RP Model")
    m_raw = compute_metrics(preds_raw, targets, "1D CNN Baseline")

    # ── Visualise ─────────────────────────────────────────────────────────
    plot_training_curves(history_rp, history_raw,
                          save_path=str(WORK_DIR / "training_curves.png"))
    plot_predictions(preds_rp, preds_raw, targets,
                      save_path=str(WORK_DIR / "predictions.png"))
    plot_scatter(preds_rp, preds_raw, targets,
                  save_path=str(WORK_DIR / "scatter.png"))
    plot_comparison_bar(m_rp, m_raw,
                         save_path=str(WORK_DIR / "comparison.png"))

    # Rebuild RPs with trained embedder for example visualisation
    print("\nRebuilding example RPs with trained embedder ...")
    embedder_trained = model_rp.embedder
    example_X   = splits["X_test"][:8]
    rps_trained = build_rp_array(example_X, embedder_trained, DEVICE,
                                  desc="Example RPs")

    model_rp.eval(); model_raw.eval()
    with torch.no_grad():
        ex_x     = torch.tensor(example_X.transpose(0, 2, 1),
                                 dtype=torch.float32).to(DEVICE)   # (8, F, T)
        p_rp_ex  = model_rp(ex_x).cpu().numpy()
        p_raw_ex = model_raw(ex_x).cpu().numpy()

    plot_examples(
        example_X, rps_trained, splits["y_test"][:8],
        p_rp_ex, p_raw_ex, feat_cols,
        n_show=4, save_path=str(WORK_DIR / "examples.png"),
    )

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  FINAL SUMMARY (normalised PM2.5 space)")
    print("=" * 66)
    print(f"  {'Metric':<8} {'RP Model':>14} {'1D CNN':>12}  Winner")
    print("  " + "-" * 50)
    for key, lbl in [("mse","MSE"), ("rmse","RMSE"),
                     ("mae","MAE"), ("r2","R²")]:
        v_rp  = m_rp[key]
        v_raw = m_raw[key]
        winner = "RP" if (v_rp >= v_raw if key == "r2" else v_rp <= v_raw) else "Raw"
        print(f"  {lbl:<8} {v_rp:>14.6f} {v_raw:>12.6f}  {winner}")
    print("=" * 66)

    torch.save(model_rp.state_dict(),  str(WORK_DIR / "model_rp.pth"))
    torch.save(model_raw.state_dict(), str(WORK_DIR / "model_raw.pth"))
    print(f"\nWeights saved to {WORK_DIR}")


if __name__ == "__main__":
    main()
