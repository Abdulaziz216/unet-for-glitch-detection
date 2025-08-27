"""
Modular TOD → focal-plane → model → detector-hits pipeline
Author: Anas

This file collects the building blocks into a clean API. It reuses
existing FocalPlane/FocalPlaneFlex + filtering + TinyUNet implementations,
then organizes them into small, testable functions.

"""
from __future__ import annotations

from dataclasses import dataclass, asdict  # asdict unused but kept for API stability
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.interpolate import RegularGridInterpolator

# Optional runtime import of cutslib; we gate usages so static tooling won't fail
try:
    import cutslib as cl  # type: ignore
except Exception as _e:  # noqa: BLE001
    cl = None  # will assert at runtime where needed

# -----------------------------------------------------------------------------
# Units & small helpers
# -----------------------------------------------------------------------------
deg: float = np.deg2rad(1.0)
arcmin: float = deg / 60.0
Hz: int = 1


# -----------------------------------------------------------------------------
# Model: TinyUNet 
# -----------------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    """Conv → BatchNorm → ReLU block used throughout TinyUNet.

    Args:
        in_c: Number of input channels.
        out_c: Number of output channels.
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    """A minimal U-Net-style encoder-decoder for 2D segmentation.

    Designed for single-channel inputs (1xHxW) and C output classes.

    Args:
        n_classes: Number of output classes (channels in the logits).
        base_c: Base channel width for the first stage.
        p_drop: Dropout probability in decoder blocks.

    Shape:
        Input:  (B, 1, H, W)
        Output: (B, n_classes, H, W) logits (apply sigmoid outside if needed)
    """

    def __init__(self, n_classes: int = 3, base_c: int = 32, p_drop: float = 0.25):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(ConvBNReLU(1, base_c), ConvBNReLU(base_c, base_c))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base_c, base_c * 2))
        # Bottleneck
        self.bott = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base_c * 2, base_c * 4))
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBNReLU(base_c * 4, base_c * 2), nn.Dropout2d(p_drop))
        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBNReLU(base_c * 2, base_c), nn.Dropout2d(p_drop))
        self.outc = nn.Conv2d(base_c, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns raw logits shaped (B, n_classes, H, W). Use `torch.sigmoid` for
        multilabel problems or `torch.softmax` for mutually-exclusive classes.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bott(e2)
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], 1))  # skip-connection from enc2
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], 1))  # skip-connection from enc1
        return self.outc(d2)  # logits


# -----------------------------------------------------------------------------
# Pre/post helpers
# -----------------------------------------------------------------------------

def pct_clip_norm(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile clip then normalize to [0, 1].

    Useful to stabilize dynamic range across slices before model inference or plotting.

    Args:
        img: Input image array.
        lo_pct: Lower percentile for clipping.
        hi_pct: Upper percentile for clipping.
    Returns:
        Float array scaled to [0, 1].
    """
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    img2 = np.clip(img, lo, hi)
    return (img2 - lo) / (hi - lo + 1e-12)


def highpass_rows(data: np.ndarray, srate: float, fc: float = 0.5, order: int = 5) -> np.ndarray:
    """High-pass filter each row (detector) independently using a Butterworth SOS.

    Args:
        data: Array with shape (D, T) or (rows, time).
        srate: Sampling rate in Hz.
        fc: Cutoff frequency in Hz.
        order: Filter order.
    Returns:
        Same shape as `data`, filtered along axis=1.
    """
    sos = sig.butter(order, fc, btype="highpass", fs=srate, output="sos")
    return sig.sosfiltfilt(sos, data, axis=1, padtype="odd", padlen=3 * order)


def composite_map(probs: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """Compose per-class probability maps into a quick RGB overlay for visualization.

    Args:
        probs: Tensor (C, H, W) with per-class probabilities or scores in [0,1].
        thr: Threshold above which a pixel is considered active for coloring.
    Returns:
        uint8 RGB image (H, W, 3).
    """
    masks = probs > thr
    colors = torch.tensor(
        [
            [220, 20, 60],   # CR (crimson)
            [255, 105, 180], # PS (hot pink)
            [255, 255, 0],   # EL (yellow)
        ],
        dtype=torch.uint8,
    )
    C, H, W = probs.shape
    comp = torch.zeros(3, H, W, dtype=torch.uint8)
    for i in range(C):
        comp[:, masks[i]] = colors[i].view(3, 1)
    return comp.permute(1, 2, 0).cpu().numpy()


# -----------------------------------------------------------------------------
# Focal plane classes 
# -----------------------------------------------------------------------------
@dataclass
class FocalPlane:
    """Sparse set of detector coordinates (radians) on the focal plane.

    Attributes:
        x: 1D array of detector x-coordinates in radians.
        y: 1D array of detector y-coordinates in radians.
    """

    x: np.ndarray  # radians
    y: np.ndarray

    @classmethod
    def from_radius(cls, radius: float = 0.5 * deg, nrows: int = 30) -> "FocalPlane":
        """Generate a synthetic circular focal plane for demos/tests.

        Creates an nrowsxnrows grid and keeps points inside the unit circle,
        then scales to the given radial extent.
        """
        X, Y = np.meshgrid(np.linspace(-1, 1, nrows), np.linspace(-1, 1, nrows))
        m = (X**2 + Y**2) < 1
        return cls(X[m] * radius, Y[m] * radius)

    def get_circular_cover(self, n_dummy: int = 50) -> "FocalPlane":
        """Return points forming a circle that encloses the current detectors.

        Helpful for triangulation padding and plotting extents.
        """
        if self.x.size == 0:
            return FocalPlane(np.array([]), np.array([]))
        cx, cy = float(self.x.mean()), float(self.y.mean())
        r = float(np.sqrt(((self.x - cx) ** 2 + (self.y - cy) ** 2)).max()) if self.x.size else 0.0
        ang = np.linspace(0, 2 * np.pi, n_dummy, endpoint=False)
        return FocalPlane(cx + r * np.cos(ang), cy + r * np.sin(ang))

    @property
    def n_dets(self) -> int:
        """Number of detectors (points)."""
        return int(self.x.size)


class FocalPlaneFlex:
    """Triangulation-based resampling between sparse detectors and a dense grid.

    This class precomputes barycentric weights on a Delaunay triangulation of the
    detector coordinates. It provides fast mapping both ways:
    - `tod_to_video`: (D, T) → (T, H, W)
    - `video_to_tod`: (T, H, W) → (T, D)

    Args:
        fplane: FocalPlane describing (x, y) detector locations in radians.
        grid_resolution: Output grid size H=W=grid_resolution.
        verbose: If True, prints triangulation/weight stats.
    """

    def __init__(self, fplane: FocalPlane, grid_resolution: int = 50, verbose: bool = False):
        self.fplane = fplane
        self.grid_resolution = grid_resolution
        self.verbose = verbose

        # (D, 2) array of detector coordinates in the plane
        self.detector_coords = np.column_stack((self.fplane.x, self.fplane.y))
        self.triangulation = Delaunay(self.detector_coords)

        # Precompute target grid and interpolation weights once
        self.output_grid, self.x_grid, self.y_grid, self.grid_shape = self._create_output_grid()
        self.weight_matrix, self.valid_mask = self._precompute_weights()

        if self.verbose:
            print(
                f"Triangulation: {len(self.triangulation.simplices)} triangles, "
                f"weights nnz={self.weight_matrix.nnz}, valid={int(self.valid_mask.sum())}/{len(self.valid_mask)}"
            )

    def _create_output_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """Build a regular HxW grid that covers the detector convex hull with padding."""
        pad = 0.1 * (np.max(self.fplane.x) - np.min(self.fplane.x))
        x_min, x_max = np.min(self.fplane.x) - pad, np.max(self.fplane.x) + pad
        y_min, y_max = np.min(self.fplane.y) - pad, np.max(self.fplane.y) + pad
        x_grid = np.linspace(x_min, x_max, self.grid_resolution)
        y_grid = np.linspace(y_min, y_max, self.grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack((X.flatten(), Y.flatten()))
        return grid_points, x_grid, y_grid, (self.grid_resolution, self.grid_resolution)

    def _precompute_weights(self) -> Tuple[coo_matrix, np.ndarray]:
        """Compute barycentric weights from detector vertices to each output grid point.

        Returns a sparse matrix W (N_grid x D) where each row contains the 3 barycentric
        weights for the triangle that encloses the grid point; `valid_mask` marks which
        grid points are inside the triangulation.
        """
        n_grid_points = len(self.output_grid)
        rows: List[int] = []
        cols: List[int] = []
        weights: List[float] = []
        valid_mask = np.zeros(n_grid_points, dtype=bool)

        simplex_indices = self.triangulation.find_simplex(self.output_grid)  # -1 means outside
        for grid_idx, simplex_idx in enumerate(simplex_indices):
            if simplex_idx == -1:
                continue
            valid_mask[grid_idx] = True
            triangle_vertices = self.triangulation.simplices[simplex_idx]
            bary = self._barycentric(self.output_grid[grid_idx], self.detector_coords[triangle_vertices])
            for i, vtx in enumerate(triangle_vertices):
                w = float(bary[i])
                if w > 1e-10:  # ignore near‑zero numerical noise
                    rows.append(grid_idx)
                    cols.append(int(vtx))
                    weights.append(w)

        weight_matrix = coo_matrix((weights, (rows, cols)), shape=(n_grid_points, len(self.detector_coords)))
        return weight_matrix, valid_mask

    @staticmethod
    def _barycentric(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
        """Compute barycentric coordinates of `point` relative to triangle `tri`.

        Args:
            point: (2,) target XY point.
            tri: (3, 2) triangle vertex coordinates.
        Returns:
            (3,) array of weights (w, v, u) that sum to 1.
        """
        A, B, C = tri
        v0 = C - A
        v1 = B - A
        v2 = point - A
        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot02 = float(np.dot(v0, v2))
        dot11 = float(np.dot(v1, v1))
        dot12 = float(np.dot(v1, v2))
        inv = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        w = 1.0 - u - v
        return np.array([w, v, u], dtype=float)

    def tod_to_video(self, tod_data: np.ndarray) -> np.ndarray:
        """Map TOD (detectorxtime) to a sequence of images (timexHxW).

        Accepts shapes (D, T) or (T, D) or (T,) (which is reshaped to (1, T)).
        If input is (D, T) it is transposed to (T, D) internally. The number of
        detectors D must match `len(self.fplane.x)`.
        """
        if tod_data.ndim == 1:
            tod_data = tod_data.reshape(1, -1)
        if tod_data.shape[0] == len(self.fplane.x):
            tod_data = tod_data.T  # (T, D)
        if tod_data.shape[1] != len(self.fplane.x):
            raise ValueError(
                f"tod_data shape {tod_data.shape} does not match n_dets {len(self.fplane.x)}"
            )
        # Multiply sparse weights: (N_grid × D) @ (D × T) → (N_grid × T) then T×N_grid
        video_flat = self.weight_matrix.dot(tod_data.T).T
        # Zero out points outside triangulation
        video_flat[:, ~self.valid_mask] = 0
        n_frames = video_flat.shape[0]
        H, W = self.grid_shape
        return video_flat.reshape(n_frames, H, W)

    def video_to_tod(self, video_data: np.ndarray) -> np.ndarray:
        """Map a single frame or stack of frames (HxW) or (TxHxW) back to (TxD).

        Uses `RegularGridInterpolator` on the (y, x) axes to sample detector coords.
        Returns (T, D) if multiple frames, else (D,) for a single frame.
        """
        if video_data.ndim == 2:
            video_data = video_data[np.newaxis, :, :]
        n_frames, h, w = video_data.shape
        if (h, w) != self.grid_shape:
            raise ValueError(f"Input video shape {(h, w)} must match grid {self.grid_shape}")
        query = self.detector_coords
        n_det = len(query)
        tod = np.zeros((n_frames, n_det), dtype=float)
        for i in range(n_frames):
            frame = video_data[i]
            # Interpolator takes axes in (y, x) order
            interp = RegularGridInterpolator((self.y_grid, self.x_grid), frame, method="linear", bounds_error=False, fill_value=0)
            tod[i, :] = interp(query[:, ::-1])  # query expects (y, x)
        return tod if n_frames > 1 else tod.flatten()


# -----------------------------------------------------------------------------
# Contexts & data contracts
# -----------------------------------------------------------------------------
@dataclass
class TodContext:
    """Container for a loaded/processed TOD and per-detector metadata.

    Attributes:
        tod_id: String identifier for the TOD.
        depot: Path to data depot used by cutslib.
        release: Data release tag.
        band: Frequency band (e.g., 'f090').
        s_rate_hz: Sampling rate in Hz.
        data: Array (D, T) after `cl.quick_transform` and band selection.
        det_uids: Detector IDs (D,).
        sky_x, sky_y: Detector coordinates in radians (D,).
        data_hp: Optional cached high-pass filtered data (D, T).
    """

    tod_id: str
    depot: str
    release: str
    band: str
    s_rate_hz: float
    data: np.ndarray            # full (D,T) after cl.quick_transform + band + uncut
    det_uids: np.ndarray        # (D,)
    sky_x: np.ndarray           # radians (D,)
    sky_y: np.ndarray           # radians (D,)
    data_hp: Optional[np.ndarray] = None  # cached high-pass (D,T)


@dataclass
class FpContext:
    """Bundle for focal-plane geometry and cached resampling artifacts."""

    grid_res: int
    fp: FocalPlane
    flex: FocalPlaneFlex
    video_cache: Dict[Tuple[float, int], np.ndarray]  # key=(fc_hz, order)


@dataclass
class DetHit:
    """Per-detector hit with probabilities and labels crossing thresholds."""

    det_uid: int
    sky_x_rad: float
    sky_y_rad: float
    probs: Dict[str, float]
    labels: List[str]


@dataclass
class SliceResult:
    """Result for a single time slice: detector hits + meta describing inputs."""

    slice_from: int
    slice_to: int
    hits: List[DetHit]
    meta: Dict[str, object]


SliceReduce = Literal["max", "sum", "mean"]


# -----------------------------------------------------------------------------
# IO / contexts
# -----------------------------------------------------------------------------

def _assert_cutslib():
    """Raise a helpful error if `cutslib` isn't importable in this environment."""
    if cl is None:
        raise RuntimeError("cutslib is not importable in this environment.")


def load_tod_ctx(
    tod_id: str,
    *,
    depot: str,
    release: str,
    band: str = "f090",
    transform_steps: Sequence[str] = ("ff_mce", "cal", "demean", "detrend"),
    s_rate_hz: Optional[float] = None,
) -> TodContext:
    """Load a TOD via cutslib and build a `TodContext` for downstream steps.

    Performs common transforms, selects uncut detectors for a given band, and
    extracts per-detector metadata (uid, sky_x, sky_y). Sampling rate is taken
    from the TOD if not specified.
    """
    _assert_cutslib()
    tod = cl.load_tod(tod_id, depot=depot, autoloads=["cuts", "partial", "cal"], release=release)
    cl.quick_transform(tod, steps=list(transform_steps))

    uncut = tod.cuts.get_uncut()
    arr_f = np.asarray(tod.info.array_data["fcode"])  # e.g., 'f090'
    fmask = np.where(arr_f == band)[0]
    keep = np.intersect1d(uncut, fmask)
    if keep.size == 0:
        raise ValueError(f"No uncut detectors for band '{band}' in {tod_id}")

    data = np.asarray(tod.data[keep, :])
    sky_x = np.deg2rad(np.asarray(tod.info.array_data["sky_x"][keep]))
    sky_y = np.deg2rad(np.asarray(tod.info.array_data["sky_y"][keep]))
    det_uids = np.asarray(tod.info.array_data["det_uid"][keep])

    # Sampling rate: prefer provided else try tod.info
    if s_rate_hz is None:
        # fall back: many cutslib TODs have sample rate under .info
        s_rate_hz = float(getattr(tod.info, "srate", 400.0))

    return TodContext(
        tod_id=tod_id,
        depot=depot,
        release=release,
        band=band,
        s_rate_hz=float(s_rate_hz),
        data=data,
        det_uids=det_uids,
        sky_x=sky_x,
        sky_y=sky_y,
    )


def build_fplane(ctx: TodContext, grid_res: int = 32, *, verbose: bool = False) -> FpContext:
    """Construct focal-plane geometry + resampler for a given TOD context."""
    fp = FocalPlane(ctx.sky_x, ctx.sky_y)
    flex = FocalPlaneFlex(fp, grid_resolution=grid_res, verbose=verbose)
    return FpContext(grid_res=grid_res, fp=fp, flex=flex, video_cache={})


# -----------------------------------------------------------------------------
# Preprocess → video & slice images
# -----------------------------------------------------------------------------

def tod_to_video(
    ctx: TodContext,
    fpctx: FpContext,
    *,
    hp_filter: Optional[Tuple[float, int]] = (2.0, 5),  # (fc_hz, order)
    use_cache: bool = True,
) -> np.ndarray:
    """Convert high-pass-filtered TOD to a (T, H, W) video on the focal-plane grid.

    Caches results per (cutoff, order) to avoid re-computing when iterating.

    Args:
        ctx: TOD context with data and metadata.
        fpctx: Focal-plane/resampler context.
        hp_filter: Optional high-pass filter tuple (fc_hz, order). If None, raw data used.
        use_cache: Reuse previously computed video for same HP filter.
    Returns:
        Array of shape (T, H, W).
    """
    fc_hz, order = (hp_filter if hp_filter is not None else (None, None))

    # Cache key includes filter + grid resolution
    cache_key = (float(fc_hz) if fc_hz is not None else -1.0, int(order) if order is not None else -1)
    if use_cache and hp_filter is not None and cache_key in fpctx.video_cache:
        return fpctx.video_cache[cache_key]

    data = ctx.data
    if hp_filter is not None:
        if ctx.data_hp is None or ctx.data_hp.shape != ctx.data.shape or ctx.s_rate_hz is None:
            ctx.data_hp = highpass_rows(ctx.data, srate=ctx.s_rate_hz, fc=float(fc_hz), order=int(order))
        data = ctx.data_hp

    video = fpctx.flex.tod_to_video(data)
    if use_cache and hp_filter is not None:
        fpctx.video_cache[cache_key] = video
    return video


def _normalize_slice(sl: Union[slice, Tuple[int, int]]) -> Tuple[int, int]:
    """Coerce `slice` or `(start, stop)` into a validated `(start, stop)` pair."""
    if isinstance(sl, tuple):
        start, stop = int(sl[0]), int(sl[1])
    else:
        start, stop = int(sl.start), int(sl.stop)
    if stop <= start:
        raise ValueError(f"Invalid slice: {sl}")
    return start, stop


def slices_to_images(
    video: np.ndarray,
    slices: Sequence[Union[slice, Tuple[int, int]]],
    *,
    reduce: SliceReduce = "max",
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = pct_clip_norm,
) -> List[np.ndarray]:
    """Reduce time ranges of a video (T, H, W) into 2D images via max/sum/mean.

    Optionally normalize each reduced image for consistent dynamic range.
    """
    images: List[np.ndarray] = []
    for sl in slices:
        s0, s1 = _normalize_slice(sl)
        cube = video[s0:s1]
        if cube.size == 0:
            raise ValueError(f"Slice {sl} is empty against video length {len(video)}")
        if reduce == "max":
            img = cube.max(axis=0)
        elif reduce == "sum":
            img = cube.sum(axis=0)
        elif reduce == "mean":
            img = cube.mean(axis=0)
        else:
            raise ValueError(f"Unknown reduce: {reduce}")
        if normalize is not None:
            img = normalize(img)
        images.append(img.astype(np.float32))
    return images


# -----------------------------------------------------------------------------
# Inference & mapping back
# -----------------------------------------------------------------------------

def run_model(
    model: nn.Module,
    device: Union[str, torch.device],
    images: Sequence[np.ndarray],
    *,
    batch_size: int = 32,
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    """Run the segmentation model on a list of 2D images.

    Args:
        model: PyTorch module producing (B, C, H, W) logits.
        device: Torch device or string (e.g., "cuda", "cpu").
        images: Sequence of (H, W) float arrays.
        batch_size: Inference batch size.
        apply_sigmoid: If True, applies `sigmoid` to logits and returns probabilities.
    Returns:
        Tensor of shape (N, C, H, W) on CPU.
    """
    device = torch.device(device)
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = np.stack(images[i : i + batch_size], axis=0)
            t = torch.from_numpy(batch).unsqueeze(1).to(device)  # (B,1,H,W)
            logits = model(t)
            probs = torch.sigmoid(logits) if apply_sigmoid else logits
            outs.append(probs.detach().cpu())
    return torch.cat(outs, dim=0)  # (N,C,H,W)


def preds_to_det_hits(
    fpctx: FpContext,
    preds_for_one: torch.Tensor,  # (C,H,W) on CPU
    *,
    class_names: Sequence[str] = ("CR", "PS", "EL"),
    thr_per_class: Optional[Dict[str, float]] = None,
    det_uids: np.ndarray,
    sky_x: np.ndarray,
    sky_y: np.ndarray,
) -> List[DetHit]:
    """Convert per-pixel class probabilities back to per-detector hits.

    Steps:
      1) Map each per-class 2D map (H, W) → (D,) using `video_to_tod`.
      2) Threshold per detector and collect labels above class-specific thresholds.

    Args:
        fpctx: Focal-plane/resampler context.
        preds_for_one: Probabilities for a single image (C, H, W) on CPU.
        class_names: Names aligned with channels (C order).
        thr_per_class: Dict of thresholds per class name.
        det_uids, sky_x, sky_y: Per-detector metadata.
    Returns:
        List of `DetHit` for detectors with ≥1 active class.
    """
    if thr_per_class is None:
        thr_per_class = {"CR": 0.5, "PS": 0.5, "EL": 0.5}
    # Map each class prob back to detectors
    C, H, W = preds_for_one.shape
    probs_np = preds_for_one.numpy()
    det_probs = []  # (D,C)
    for c in range(C):
        tod_vals = fpctx.flex.video_to_tod(probs_np[c])  # (D,)
        det_probs.append(tod_vals)
    det_probs = np.stack(det_probs, axis=1)  # (D,C)

    # Determine labels by per-class thresholds
    hits: List[DetHit] = []
    for d in range(det_probs.shape[0]):
        p = det_probs[d]
        labels = [cls for cls, pv in zip(class_names, p) if pv >= thr_per_class.get(cls, 0.5)]
        if not labels:
            continue
        hits.append(
            DetHit(
                det_uid=int(det_uids[d]),
                sky_x_rad=float(sky_x[d]),
                sky_y_rad=float(sky_y[d]),
                probs={cls: float(pv) for cls, pv in zip(class_names, p)},
                labels=labels,
            )
        )
    return hits


def detect_on_slices(
    ctx: TodContext,
    fpctx: FpContext,
    model: nn.Module,
    device: Union[str, torch.device],
    slices: Sequence[Union[slice, Tuple[int, int]]],
    *,
    reduce: SliceReduce = "max",
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = pct_clip_norm,
    hp_filter: Optional[Tuple[float, int]] = (2.0, 5),
    thr_per_class: Optional[Dict[str, float]] = None,
    batch_size: int = 32,
    class_names: Sequence[str] = ("CR", "PS", "EL"),
) -> List[SliceResult]:
    """End-to-end detection on time slices: TOD → video → images → probs → hits.

    Returns one `SliceResult` per slice with the corresponding per-detector hits
    and a metadata dict capturing all key configuration knobs.
    """
    video = tod_to_video(ctx, fpctx, hp_filter=hp_filter, use_cache=True)
    images = slices_to_images(video, slices, reduce=reduce, normalize=normalize)
    probs = run_model(model, device, images, batch_size=batch_size, apply_sigmoid=True)

    results: List[SliceResult] = []
    for i, sl in enumerate(slices):
        s0, s1 = _normalize_slice(sl)
        hits = preds_to_det_hits(
            fpctx,
            probs[i],
            class_names=class_names,
            thr_per_class=thr_per_class,
            det_uids=ctx.det_uids,
            sky_x=ctx.sky_x,
            sky_y=ctx.sky_y,
        )
        meta = {
            "tod_id": ctx.tod_id,
            "release": ctx.release,
            "depot": ctx.depot,
            "band": ctx.band,
            "grid_res": fpctx.grid_res,
            "hp_filter": hp_filter,
            "reduce": reduce,
            "normalize": None if normalize is None else getattr(normalize, "__name__", str(normalize)),
            "class_names": list(class_names),
            "thr_per_class": {} if thr_per_class is None else dict(thr_per_class),
        }
        results.append(SliceResult(slice_from=s0, slice_to=s1, hits=hits, meta=meta))
    return results


# -----------------------------------------------------------------------------
# Visualization utilities (each is pure; return the figure)
# -----------------------------------------------------------------------------

def plot_filtered_images(images: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None):
    """Show a row of grayscale images; returns the Matplotlib Figure."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, img in enumerate(images):
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
        if titles is not None:
            axes[0, i].set_title(titles[i])
    fig.tight_layout()
    return fig


def plot_prob_maps(prob_tensor_for_one: torch.Tensor, class_names: Sequence[str] = ("CR", "PS", "EL")):
    """Plot per-class probability maps for a single sample; returns the Figure."""
    C, H, W = prob_tensor_for_one.shape
    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4), squeeze=False)
    for j in range(C):
        axes[0, j].imshow(prob_tensor_for_one[j].cpu(), cmap="gray")
        axes[0, j].axis("off")
        axes[0, j].set_title(f"Prob {class_names[j]}")
    fig.tight_layout()
    return fig


def plot_composite(prob_tensor_for_one: torch.Tensor, thr: float = 0.5):
    """Plot the composite RGB overlay for a single sample; returns the Figure."""
    img = composite_map(prob_tensor_for_one, thr=thr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Composite (thr={thr})")
    fig.tight_layout()
    return fig


def plot_class_scatter(
    base_img: np.ndarray,
    sky_x: np.ndarray,
    sky_y: np.ndarray,
    det_probs: np.ndarray,  # (D,C)
    *,
    thr_per_class: Dict[str, float],
    class_names: Sequence[str] = ("CR", "PS", "EL"),
):
    """Scatter detector positions over `base_img`, color-coded by thresholded classes."""
    colors = {"PS": "tab:red", "CR": "tab:blue", "EL": "tab:orange"}
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(base_img, cmap="gray")
    for idx, cls in enumerate(class_names):
        mask = det_probs[:, idx] >= thr_per_class.get(cls, 0.5)
        if mask.any():
            ax.scatter(sky_x[mask], sky_y[mask], s=30, c=colors.get(cls, "k"), label=f"{cls} ({int(mask.sum())})")
    ax.set_xlabel("sky_x (rad)")
    ax.set_ylabel("sky_y (rad)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_timestreams(hp: np.ndarray, sl: Tuple[int, int], masks: Dict[str, np.ndarray]):
    """Plot raw/high-pass timestreams for detectors that cross per-class thresholds.

    Panels: PS, CR, and (optionally) EL if any EL detections are present.

    Args:
        hp: High-pass filtered data with shape (D, T).
        sl: (start, stop) time indices for the slice to visualize.
        masks: Dict with boolean masks per class over detectors.
    Returns:
        Matplotlib Figure.
    """
    s0, s1 = sl

    el_present = (masks.get("EL", None) is not None) and masks["EL"].any()
    ncols = 3 if el_present else 2
    fig_w = 12 if ncols == 2 else 18

    fig = plt.figure(figsize=(fig_w, 4.2))

    # Left: PS 
    ax1 = fig.add_subplot(1, ncols, 1)
    ax1.plot(hp[:, s0:s1].T, alpha=0.15, color="gray")
    if masks.get("PS", None) is not None and masks["PS"].any():
        ax1.plot(hp[masks["PS"], s0:s1].T, alpha=0.7, color="tab:pink")
    ax1.set_title(f"Slice {s0}:{s1} PS-only traces", fontsize = 16)
    ax1.set_xlabel("Sample idx", fontsize = 14)
    ax1.set_ylabel("Signal", fontsize = 14)

    # Middle: CR 
    ax2 = fig.add_subplot(1, ncols, 2)
    ax2.plot(hp[:, s0:s1].T, alpha=0.15, color="gray")
    if masks.get("CR", None) is not None and masks["CR"].any():
        ax2.plot(hp[masks["CR"], s0:s1].T, alpha=0.7, color="tab:red")
    ax2.set_title(f"Slice {s0}:{s1} CR-only traces", fontsize = 16)
    ax2.set_xlabel("Sample idx", fontsize = 14)
    ax2.set_ylabel("Signal", fontsize = 14)

    # Right: EL (only if present)
    if el_present:
        ax3 = fig.add_subplot(1, ncols, 3)
        ax3.plot(hp[:, s0:s1].T, alpha=0.15, color="gray")
        ax3.plot(hp[masks["EL"], s0:s1].T, alpha=0.7, color="gold")  # visible yellow
        ax3.set_title(f"Slice {s0}:{s1} EL-only traces")
        ax3.set_xlabel("Sample idx")
        ax3.set_ylabel("HP signal")

    fig.tight_layout()
    return fig



# -----------------------------------------------------------------------------
# Demo wrapper (optional) – convenient notebook/gallery function
# -----------------------------------------------------------------------------

def demo_gallery(
    ctx: TodContext,
    fpctx: FpContext,
    model: nn.Module,
    device: Union[str, torch.device],
    slices: Sequence[Union[slice, Tuple[int, int]]],
    *,
    reduce: SliceReduce = "max",
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = pct_clip_norm,
    hp_filter: Optional[Tuple[float, int]] = (2.0, 5),
    thr_per_class: Optional[Dict[str, float]] = None,
    show_filtered: bool = True,
):
    """One-stop visual gallery over a list of slices.

    Uses the same pipeline primitives and RETURNS the structured `SliceResult`
    list so you can programmatically inspect hits in addition to the figures.
    """
    # Build video/images and run model once
    video  = tod_to_video(ctx, fpctx, hp_filter=hp_filter, use_cache=True)
    images = slices_to_images(video, slices, reduce=reduce, normalize=normalize)
    probs  = run_model(model, device, images)

    # Optional: show the filtered slice images
    if show_filtered:
        _ = plot_filtered_images(images, [f"slice {i}" for i, _ in enumerate(slices)])
        plt.show()

    results: List[SliceResult] = []
    class_names = ("CR", "PS", "EL")
    tpc = {"CR": 0.5, "PS": 0.5, "EL": 0.5} if thr_per_class is None else thr_per_class

    for i, sl in enumerate(slices):
        s0, s1 = _normalize_slice(sl)

        # Composite & probability maps
        _ = plot_composite(probs[i], thr=float(np.mean(list(tpc.values()))))
        plt.show()
        _ = plot_prob_maps(probs[i])
        plt.show()

        # For scatter/timestreams we back-project per-class prob maps → (D,C)
        C = probs[i].shape[0]
        det_probs = []
        for c in range(C):
            det_probs.append(fpctx.flex.video_to_tod(probs[i, c].numpy()))
        det_probs = np.stack(det_probs, axis=1)

        # Class-colored scatter on the focal plane (with correct extent)
        hits_i = preds_to_det_hits(
            fpctx, probs[i],
            det_uids=ctx.det_uids, sky_x=ctx.sky_x, sky_y=ctx.sky_y,
            thr_per_class=thr_per_class, class_names=("CR","PS","EL")
        )
        _ = plot_class_scatter_hits(
            images[i], fpctx, hits_i, title=f"Slice {s0}:{s1}"
        )
        plt.show()

        # Timestream sanity (PS/CR panels)
        masks = {
            "PS": det_probs[:, 1] >= tpc.get("PS", 0.5),
            "CR": det_probs[:, 0] >= tpc.get("CR", 0.5),
        }
        hp = ctx.data_hp if ctx.data_hp is not None else (
            highpass_rows(ctx.data, ctx.s_rate_hz, fc=hp_filter[0], order=hp_filter[1]) if hp_filter else ctx.data
        )
        _ = plot_timestreams(hp, (s0, s1), masks)
        plt.show()

        # Build hits for this slice (same as pipeline) and accumulate results
        hits = preds_to_det_hits(
            fpctx,
            probs[i],
            class_names=class_names,
            thr_per_class=tpc,
            det_uids=ctx.det_uids,
            sky_x=ctx.sky_x,
            sky_y=ctx.sky_y,
        )
        meta = {
            "tod_id": ctx.tod_id,
            "release": ctx.release,
            "depot": ctx.depot,
            "band": ctx.band,
            "grid_res": fpctx.grid_res,
            "hp_filter": hp_filter,
            "reduce": reduce,
            "normalize": None if normalize is None else getattr(normalize, "__name__", str(normalize)),
            "class_names": list(class_names),
            "thr_per_class": dict(tpc),
        }
        results.append(SliceResult(slice_from=s0, slice_to=s1, hits=hits, meta=meta))

    return results



# -----------------------------------------------------------------------------
# Convenience: DataFrame/JSON export (optional)
# -----------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


def results_to_dataframe(results: Sequence[SliceResult]) -> "pd.DataFrame":
    """Flatten a list of `SliceResult` into a tidy pandas DataFrame.

    Columns include slice bounds, detector metadata, per-class probabilities
    (prefixed with `prob_`), and a comma-joined `labels` string.
    """
    if pd is None:
        raise RuntimeError("pandas is not available")
    rows = []
    for r in results:
        for h in r.hits:
            rows.append(
                {
                    "slice_from": r.slice_from,
                    "slice_to": r.slice_to,
                    "det_uid": h.det_uid,
                    "sky_x_rad": h.sky_x_rad,
                    "sky_y_rad": h.sky_y_rad,
                    **{f"prob_{k}": v for k, v in h.probs.items()},
                    "labels": ",".join(h.labels),
                }
            )
    return pd.DataFrame(rows)

# def plot_class_scatter_fp(base_img, fpctx, sky_x, sky_y, det_probs, thr_per_class, class_names=("CR","PS","EL")):
#     colors = {"PS": "tab:red", "CR": "tab:blue", "EL": "tab:orange"}
#     xg, yg = fpctx.flex.x_grid, fpctx.flex.y_grid
#     fig, ax = plt.subplots(figsize=(5.2,5.2))
#     ax.imshow(base_img, cmap="gray", extent=(xg[0], xg[-1], yg[-1], yg[0]))
#     for idx, cls in enumerate(class_names):
#         m = det_probs[:, idx] >= thr_per_class.get(cls, 0.5)
#         if m.any():
#             ax.scatter(sky_x[m], sky_y[m], s=30, c=colors.get(cls, "k"), label=f"{cls} ({int(m.sum())})")
#     ax.set_xlabel("sky_x (rad)"); ax.set_ylabel("sky_y (rad)"); ax.legend(loc="upper right")
#     fig.tight_layout(); return fig

def plot_class_scatter_hits(
    base_img,
    fpctx,
    hits,                                 # list[DetHit]
    class_order=("CR","PS","EL"),
    colors={"PS": "tab:pink", "CR": "tab:red", "EL": "gold"},  # was 'tab:yellow'
    title=None,
):
    """Scatter convenience using `DetHit` objects instead of raw arrays.

    Draws on top of `base_img` with proper extent from the resampler context.
    """
    xg, yg = fpctx.flex.x_grid, fpctx.flex.y_grid
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(base_img, cmap="gray", extent=(xg[0], xg[-1], yg[-1], yg[0]))

    for cls in class_order:
        xs = [h.sky_x_rad for h in hits if cls in h.labels]
        ys = [h.sky_y_rad for h in hits if cls in h.labels]
        if xs:
            ax.scatter(xs, ys, s=30, c=colors.get(cls, "k"), label=f"{cls} ({len(xs)})")

    ax.set_xlabel("sky_x (rad)")
    ax.set_ylabel("sky_y (rad)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig



def per_slice_det_probs(fpctx, probs_one):  # (C,H,W) → (D,C)
    """Helper: back-project a single (C, H, W) probability tensor to (D, C)."""
    C = probs_one.shape[0]
    det_probs = []
    for c in range(C):
        det_probs.append(fpctx.flex.video_to_tod(probs_one[c].numpy()))
    return np.stack(det_probs, axis=1)



# Notes:
# - This module does not execute anything on import.
# - In your notebook/script, stitch like:
#   ctx  = load_tod_ctx(tod_id, depot=DEPOT_PATH, release=RELEASE_TAG, band="f090", s_rate_hz=S_RATE_HZ)
#   fp   = build_fplane(ctx, grid_res=32)
#   res  = detect_on_slices(ctx, fp, model, DEVICE, slices, reduce="max", thr_per_class={"CR":0.5,"PS":0.5,"EL":0.5})
#   df   = results_to_dataframe(res)
