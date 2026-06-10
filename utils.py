# Copyright (c) 2026 Gabriel Pascoli
# Released under the MIT Licence — see LICENSE in the root of this repository.
#
# Third-party dependencies and their licences:
#   Cellpose  (Stringer et al. 2021; Pachitariu & Stringer 2022) — BSD-3-Clause
#   scikit-image — BSD-3-Clause
#   NumPy, SciPy, pandas, matplotlib — BSD-3-Clause / PSF

"""
=============================================================
  YAP/TAZ Nuclear / Cytoplasmic Intensity Analyser
  Segmentation: Cellpose  (cyto3 + nuclei models)
=============================================================

Channel layout (0-indexed):
  0 → YAP/TAZ staining  (signal of interest)
  1 → Phalloidin         (cell body / F-actin)
  2 → Hoechst            (nucleus)

Pipeline
--------
1.  Load .czi → max-intensity-project across Z (all channels)
2.  Nuclear mask   : Cellpose 'nuclei' model on Hoechst MIP
3.  Cell mask      : Cellpose 'cyto3'  model on Phalloidin MIP
                     (Hoechst used as optional nucleus guide)
4.  Per-cell ROI   : each cell gets its own cytoplasm region
                     (cell label ∖ matched nucleus label)
5.  Quantify YAP/TAZ : nuclear mean, cyto mean, N/C ratio
6.  Outputs        : overview PNG, per-cell-panel PNG, CSV

First run will download Cellpose model weights (~250 MB total)
to ~/.cellpose/models/  — internet access required once.

Usage
-----
python yap_taz_cellpose.py  image.czi
python yap_taz_cellpose.py  folder/   -o results/
python yap_taz_cellpose.py  image.czi --diameter 40 --gpu
"""

import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation, exposure

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — CZI loader  (bioio-czi → czifile fallback)
# ══════════════════════════════════════════════════════════════

def load_czi(filepath: str) -> np.ndarray:
    """
    Return (C, Y, X) float32 max-intensity projection.
    Tries bioio-czi first, then czifile as fallback.
    """
    path = Path(filepath)

    # ── bioio-czi ─────────────────────────────────────────────
    try:
        from bioio import BioImage
        from bioio_czi import Reader
        img  = BioImage(str(path), reader=Reader)
        data = img.get_image_data("CZYX", T=0)   # (C, Z, Y, X)
        print(f"  Loaded via bioio-czi  |  CZYX {data.shape}")
        return np.max(data, axis=1).astype(np.float32)
    except Exception as e:
        print(f"  bioio-czi failed ({e}), trying czifile …")

    # ── czifile fallback ──────────────────────────────────────
    import czifile
    with czifile.CziFile(str(path)) as czi:
        raw = np.squeeze(czi.asarray())
    if raw.ndim == 4:                     # C, Z, Y, X
        return np.max(raw, axis=1).astype(np.float32)
    if raw.ndim == 3:                     # C, Y, X  (single Z)
        return raw.astype(np.float32)
    raise ValueError(f"Unexpected array shape after squeeze: {raw.shape}")


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — Helpers
# ══════════════════════════════════════════════════════════════

def norm01(arr: np.ndarray) -> np.ndarray:
    """Robust [0, 1] normalisation (clip 0.5–99.5 percentile)."""
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _otsu_fallback(channel_mip: np.ndarray, min_size: int,
                   channel_name: str) -> np.ndarray:
    """Otsu + watershed fallback when Cellpose models can't be loaded."""
    print(f"  [fallback] Running Otsu+watershed for {channel_name} …")
    img = norm01(channel_mip)
    img_s = filters.gaussian(img, sigma=1.5)
    binary = img_s > filters.threshold_otsu(img_s)
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = ndimage.binary_fill_holes(binary)
    dist   = ndimage.distance_transform_edt(binary)
    from skimage.feature import peak_local_max
    coords = peak_local_max(dist, labels=binary, min_distance=10)
    seed_mask = np.zeros_like(dist, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers = measure.label(seed_mask)
    labels  = segmentation.watershed(-dist, markers, mask=binary)
    return labels.astype(np.int32)


# ══════════════════════════════════════════════════════════════
#  SECTION 2b — GPU verification
# ══════════════════════════════════════════════════════════════

def check_gpu() -> bool:
    """
    Verify that a CUDA-capable GPU is visible to PyTorch and report
    Cellpose's own GPU status.  Returns True when GPU is confirmed usable.

    Called automatically when --gpu is passed.  You can also call it
    directly at the top of a script or notebook:

        from main import check_gpu
        check_gpu()

    Fix a mismatch:
      • CPU-only PyTorch  →  reinstall with CUDA support:
            pip install torch --index-url https://download.pytorch.org/whl/cu121
      • Driver too old    →  update NVIDIA driver (≥ 525 for CUDA 12)
      • No NVIDIA GPU     →  run without --gpu (Cellpose CPU is still fast)
    """
    print("\n── GPU / CUDA check ──────────────────────────────────────────")
    available = False

    try:
        import torch
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA built-in   : {torch.version.cuda}")

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                mem   = props.total_memory / 1024 ** 3
                print(f"  Device {i} : {props.name}"
                      f"  |  {mem:.1f} GB VRAM"
                      f"  |  Compute {props.major}.{props.minor}")
            # Warm-up: allocate a tiny tensor to confirm the driver actually works
            _ = torch.zeros(1, device="cuda")
            # Let cuDNN benchmark and select the fastest convolution kernels
            # for each unique input shape (first batch is slower; all later ones gain)
            torch.backends.cudnn.benchmark = True
            print(f"  ✓  {n} CUDA device(s) confirmed — GPU mode ACTIVE")
            print(f"  ✓  cudnn.benchmark enabled")
            available = True
        else:
            print("  ✗  torch.cuda.is_available() → False")
            print("     Possible causes:")
            print("       • No NVIDIA GPU present")
            print("       • CUDA toolkit / driver not installed or version mismatch")
            print("       • CPU-only PyTorch build")
            print("     Tip: reinstall PyTorch with CUDA support:")
            print("       pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  ✗  PyTorch not installed — cannot verify GPU.")
        print("     Install: pip install torch")

    # Also ask Cellpose directly (uses torch internally)
    try:
        from cellpose import core as cp_core
        cp_gpu = cp_core.use_gpu()
        status = "✓  available" if cp_gpu else "✗  not available"
        print(f"  Cellpose GPU    : {status}")
    except Exception:
        pass

    print("──────────────────────────────────────────────────────────────\n")
    return available


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — Cellpose nuclear segmentation (Hoechst)
# ══════════════════════════════════════════════════════════════

def make_nuclear_mask(hoechst_mip: np.ndarray,
                      diameter: float | None = None,
                      gpu: bool = False,
                      min_size: int = 200,
                      batch_size: int = 8,
                      flow_threshold: float = 0.4,
                      cellprob_threshold: float = 0.0,
                      augment: bool = False,
                      tile_overlap: float = 0.1) -> np.ndarray:
    """
    Run Cellpose 'nuclei' model on the Hoechst MIP.

    Parameters
    ----------
    flow_threshold     : 0.4 default. Raise (0.6–0.9) to recover missed nuclei at
                         the cost of more false positives.
    cellprob_threshold : 0.0 default. Lower (-2 to -6) to catch dimmer nuclei.
    augment            : 4-fold rotation augmentation — slower but much better for
                         non-round nuclei.
    tile_overlap       : fraction of tile overlap (default 0.1). Raise to 0.3 if
                         nuclei at tile edges are being clipped.
    """
    try:
        from cellpose import models as cp_models
        print("  [Cellpose] Loading 'nuclei' model …")
        model = cp_models.CellposeModel(gpu=gpu, model_type="nuclei")

        img = hoechst_mip.astype(np.float32)
        masks, flows, styles = model.eval(
            img,
            diameter           = diameter,
            channels           = [0, 0],
            flow_threshold     = flow_threshold,
            cellprob_threshold = cellprob_threshold,
            min_size           = min_size,
            batch_size         = batch_size,
            augment            = augment,
            tile_overlap       = tile_overlap,
        )
        n = int(masks.max())
        print(f"  [Cellpose] Nuclei found: {n}  (flow_thr={flow_threshold}, "
              f"prob_thr={cellprob_threshold}, augment={augment})")
        return masks.astype(np.int32)

    except Exception as e:
        print(f"  [Cellpose] nuclei model failed: {e}")
        return _otsu_fallback(hoechst_mip, min_size, "nuclei").astype(np.int32)


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — Cellpose cell segmentation (Phalloidin + Hoechst)
# ══════════════════════════════════════════════════════════════

def make_cell_mask(phall_mip: np.ndarray,
                   hoechst_mip: np.ndarray,
                   diameter: float | None = None,
                   gpu: bool = False,
                   min_size: int = 500,
                   batch_size: int = 8,
                   flow_threshold: float = 0.6,
                   cellprob_threshold: float = -2.0,
                   augment: bool = True,
                   tile_overlap: float = 0.3) -> np.ndarray:
    """
    Run Cellpose 'cyto3' model on Phalloidin MIP, using Hoechst
    as the optional nucleus guide channel.

    Defaults are tuned for elongated myoblasts (C2C12):
      flow_threshold=0.6    accept less-circular flow fields from elongated cells
      cellprob_threshold=-2 catch lower-probability cells at cell edges / in clusters
      augment=True          4-fold rotation augmentation — critical for spindle shapes
      tile_overlap=0.3      high overlap so elongated cells aren't clipped at tile edges

    For round cells (e.g. epithelial) revert to flow=0.4, prob=0.0, augment=False.
    """
    try:
        from cellpose import models as cp_models
        print("  [Cellpose] Loading 'cyto3' model …")
        model = cp_models.CellposeModel(gpu=gpu, model_type="cyto3")

        p    = phall_mip.astype(np.float32)
        h    = hoechst_mip.astype(np.float32)
        img2 = np.stack([p, h], axis=-1)

        masks, flows, styles = model.eval(
            img2,
            diameter           = diameter,
            channels           = [1, 2],
            flow_threshold     = flow_threshold,
            cellprob_threshold = cellprob_threshold,
            min_size           = min_size,
            batch_size         = batch_size,
            augment            = augment,
            tile_overlap       = tile_overlap,
        )
        n = int(masks.max())
        print(f"  [Cellpose] Cells found: {n}  (flow_thr={flow_threshold}, "
              f"prob_thr={cellprob_threshold}, augment={augment})")
        return masks.astype(np.int32)

    except Exception as e:
        print(f"  [Cellpose] cyto3 model failed: {e}")
        return _otsu_fallback(phall_mip, min_size, "cells").astype(np.int32)


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — Match cell ↔ nucleus labels
# ══════════════════════════════════════════════════════════════

def match_cells_to_nuclei(cell_labels: np.ndarray,
                           nuc_labels: np.ndarray) -> dict[int, int]:
    """
    For each cell label, find the nucleus label with maximum overlap.
    Returns  {cell_id: nucleus_id}

    Vectorised: builds a (n_cells × n_nuclei) overlap matrix in one
    np.bincount call — no Python loop over individual cells.
    """
    fg = cell_labels > 0
    c_flat = cell_labels[fg].astype(np.int64)
    n_flat = nuc_labels[fg].astype(np.int64)

    # Only care about pixels that overlap a nucleus
    valid   = n_flat > 0
    c_v     = c_flat[valid]
    n_v     = n_flat[valid]

    if c_v.size == 0:
        return {}

    n_cells = int(cell_labels.max()) + 1   # include index 0 (background)
    n_nucs  = int(nuc_labels.max())  + 1

    # Encode every (cell, nuc) pixel as a single integer, then count
    pair_idx      = c_v * n_nucs + n_v
    overlap_flat  = np.bincount(pair_idx, minlength=n_cells * n_nucs)
    overlap       = overlap_flat.reshape(n_cells, n_nucs)  # (cells, nucs)

    # For each cell row, pick the nuc column with the highest overlap
    # Column 0 is background — skip it by looking at [:, 1:]
    best_nuc_offset = overlap[:, 1:].argmax(axis=1)         # 0-indexed into cols 1..
    best_nuc        = best_nuc_offset + 1                    # shift back to label space
    has_overlap     = overlap[:, 1:].max(axis=1) > 0

    return {
        int(cid): int(best_nuc[cid])
        for cid in range(1, n_cells)          # skip background (0)
        if has_overlap[cid]
    }



# ══════════════════════════════════════════════════════════════
#  SECTION 5b — Cell / nuclear shape metrics
# ══════════════════════════════════════════════════════════════

def compute_cell_shape(labels: np.ndarray, prefix: str = "") -> pd.DataFrame:
    """
    Batch shape metrics for every labelled region via one regionprops_table call.

    Columns produced (all prefixed with *prefix*):
      area_px         area in pixels
      eccentricity    0 = circle, approaching 1 = line  (key for myoblasts)
      aspect_ratio    major_axis / minor_axis
      circularity     4π·area / perimeter²  (1 = perfect circle)
      solidity        area / convex_hull_area  (1 = convex, <1 = concave / irregular)
      extent          area / bounding_box_area
      major_axis_px   length of major axis
      minor_axis_px   length of minor axis
      orientation_deg angle of major axis in degrees  (−90 to +90)

    Returns a DataFrame indexed by the integer label value.
    """
    props = measure.regionprops_table(
        labels,
        properties=[
            "label",
            "area",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "perimeter",
            "solidity",
            "extent",
        ],
    )
    df = pd.DataFrame(props)

    # Derived metrics
    minor = df["minor_axis_length"].replace(0, np.nan)
    df["aspect_ratio"] = (df["major_axis_length"] / minor).round(3)
    df["circularity"] = (
            4 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-9)
    ).clip(0, 1).round(4)

    # Explicit rename with prefix
    rename = {
        "label": "label",
        "area": f"{prefix}area_px",
        "eccentricity": f"{prefix}eccentricity",
        "aspect_ratio": f"{prefix}aspect_ratio",
        "circularity": f"{prefix}circularity",
        "solidity": f"{prefix}solidity",
        "extent": f"{prefix}extent",
        "major_axis_length": f"{prefix}major_axis_px",
        "minor_axis_length": f"{prefix}minor_axis_px",
        "orientation": f"{prefix}orientation_rad",
    }
    df = df.rename(columns=rename)

    # Orientation in degrees, then drop radians
    df[f"{prefix}orientation_deg"] = np.degrees(
        df[f"{prefix}orientation_rad"]
    ).round(1)
    df = df.drop(columns=[f"{prefix}orientation_rad"])

    return df.set_index("label")


# ══════════════════════════════════════════════════════════════
#  SECTION 5c — Cell adjacency & proximity metrics
# ══════════════════════════════════════════════════════════════

def compute_adjacency(cell_labels: np.ndarray,
                      proximity_px: int = 10) -> pd.DataFrame:
    """
    Per-cell contact and proximity metrics — all vectorised where possible.

    Columns
    -------
    n_touching        cells that share ≥1 boundary pixel with this cell
    contact_px        boundary pixels of this cell touching another cell
    contact_frac      contact_px / perimeter  (0 = isolated, 1 = fully surrounded)
    n_nearby          non-touching cells with any pixel within *proximity_px*
    nearest_dist_px   distance (px) to the nearest OTHER cell's centroid
    is_clustered      1 if n_touching > 0 or n_nearby > 0, else 0

    For C2C12 myoblasts:
      contact_frac > 0.3  →  cell is stuck between / touching neighbours
      n_touching  > 2     →  cell embedded in a cluster
      eccentricity > 0.85 + contact_frac > 0.2  →  elongated and in contact
    """
    n_cells = int(cell_labels.max())
    empty = pd.DataFrame(columns=["cell_id", "n_touching", "contact_px",
                                   "contact_frac", "n_nearby", "nearest_dist_px",
                                   "is_clustered"])
    if n_cells == 0:
        return empty

    labels32 = cell_labels.astype(np.int32)
    max_id   = n_cells + 1

    # ── 1. Contact pixels (fully vectorised) ─────────────────
    # Inner-boundary pixels of each cell that are directly adjacent to a
    # pixel belonging to a DIFFERENT non-zero cell.
    bnd = segmentation.find_boundaries(labels32, mode="inner")
    contact_mask = np.zeros_like(bnd)
    for shift, axis in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nb = np.roll(labels32, shift, axis=axis)
        contact_mask |= bnd & (nb > 0) & (nb != labels32)
    contact_px_arr = np.bincount(labels32[contact_mask].ravel(),
                                  minlength=max_id).astype(np.int32)

    # ── 2. Touching-neighbor count (vectorised) ───────────────
    # Collect unique canonical (min_id, max_id) touching pairs from
    # only 2 shift directions (right + down) to avoid duplicates.
    all_unique_pairs = []
    for shift, axis in [(1, 0), (0, 1)]:
        nb = np.roll(labels32, shift, axis=axis)
        a  = labels32.ravel()
        b  = nb.ravel()
        valid = (a > 0) & (b > 0) & (a != b)
        mn = np.minimum(a[valid], b[valid]).astype(np.int64)
        mx = np.maximum(a[valid], b[valid]).astype(np.int64)
        all_unique_pairs.append(np.unique(mn * max_id + mx))

    if all_unique_pairs:
        encoded = np.unique(np.concatenate(all_unique_pairs))
        pa = (encoded // max_id).astype(np.int32)
        pb = (encoded  % max_id).astype(np.int32)
        n_touch_arr = np.zeros(max_id, dtype=np.int32)
        np.add.at(n_touch_arr, pa, 1)
        np.add.at(n_touch_arr, pb, 1)
        # Build per-cell touching-neighbor sets for proximity exclusion
        touch_neighbors = {i: set() for i in range(max_id)}
        for a_id, b_id in zip(pa, pb):
            touch_neighbors[int(a_id)].add(int(b_id))
            touch_neighbors[int(b_id)].add(int(a_id))
    else:
        n_touch_arr   = np.zeros(max_id, dtype=np.int32)
        touch_neighbors = {i: set() for i in range(max_id)}

    # ── 3. Perimeter from regionprops ─────────────────────────
    rpt = measure.regionprops_table(labels32,
                                     properties=["label", "perimeter",
                                                 "centroid"])
    perim_arr = np.zeros(max_id, dtype=np.float64)
    perim_arr[rpt["label"]] = rpt["perimeter"]

    # ── 4. Nearest-centroid distance (vectorised cdist) ───────
    from scipy.spatial.distance import cdist
    lbl_ids   = rpt["label"].astype(int)
    centroids = np.column_stack([rpt["centroid-0"], rpt["centroid-1"]])
    if len(lbl_ids) > 1:
        dist_mat  = cdist(centroids, centroids)
        np.fill_diagonal(dist_mat, np.inf)
        min_dists = dist_mat.min(axis=1)
    else:
        min_dists = np.full(len(lbl_ids), np.nan)
    nn_dist_map = dict(zip(lbl_ids, np.round(min_dists, 1)))

    # ── 5. Nearby (non-touching) cells — loop over cells ─────
    # O(n_cells) binary_dilations; fast enough for <500 cells per field.
    n_nearby_arr = np.zeros(max_id, dtype=np.int32)
    for cid in range(1, max_id):
        cell_mask = labels32 == cid
        if not cell_mask.any():
            continue
        dilated = ndimage.binary_dilation(cell_mask, iterations=proximity_px)
        zone    = dilated & ~cell_mask
        nearby  = set(int(x) for x in np.unique(labels32[zone])
                      if x > 0 and x != cid)
        n_nearby_arr[cid] = len(nearby - touch_neighbors[cid])

    # ── assemble ──────────────────────────────────────────────
    rows = []
    for cid in range(1, max_id):
        perim  = float(perim_arr[cid])
        cpx    = int(contact_px_arr[cid])
        rows.append(dict(
            cell_id        = cid,
            n_touching     = int(n_touch_arr[cid]),
            contact_px     = cpx,
            contact_frac   = round(cpx / (perim + 1e-9), 4),
            n_nearby       = int(n_nearby_arr[cid]),
            nearest_dist_px= nn_dist_map.get(cid, float("nan")),
            is_clustered   = int(n_touch_arr[cid] > 0 or n_nearby_arr[cid] > 0),
        ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — YAP/TAZ quantification
# ══════════════════════════════════════════════════════════════

def quantify_yap(yap_mip: np.ndarray,
                 nuc_labels: np.ndarray,
                 cell_labels: np.ndarray,
                 proximity_px: int = 10,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Per-cell + whole-image nuclear / cytoplasmic YAP/TAZ stats,
    merged with cell shape, nuclear shape, and adjacency metrics.

    Background correction
    ---------------------
    The median YAP/TAZ intensity of all pixels *outside* every cell mask
    (cell_labels == 0) is computed and subtracted from every nuclear and
    cytoplasmic pixel value before computing per-cell means.  This removes
    camera/autofluorescence offset so that nuclear and cytoplasmic means
    reflect true YAP/TAZ signal rather than absolute detector counts.

    The background estimate is stored as ``yap_background_median`` in every
    output row (including the WHOLE_IMAGE summary row).

    Returns
    -------
    df_cells  : per-cell DataFrame (one row per cell with paired nucleus)
    df_all    : df_cells + WHOLE_IMAGE summary row
    ratio_map : (Y, X) float32 with N/C ratio painted per cell
    """
    cell_nuc = match_cells_to_nuclei(cell_labels, nuc_labels)

    # ── background estimation (pixels with no cell mask) ─────
    bg_mask = cell_labels == 0
    if bg_mask.any():
        yap_bg = float(np.median(yap_mip[bg_mask]))
    else:
        # Fallback: no background pixels available (very dense field)
        yap_bg = 0.0
        print("  [background] No background pixels found — skipping bg subtraction.")
    print(f"  [background] YAP/TAZ background median (outside cell mask): {yap_bg:.2f}")

    # ── batch shape metrics ───────────────────────────────────
    cell_shape = compute_cell_shape(cell_labels, prefix="")
    nuc_shape  = compute_cell_shape(nuc_labels,  prefix="nuc_")

    # ── batch centroid + area from regionprops_table ──────────
    rpt = measure.regionprops_table(
        cell_labels,
        properties=["label", "centroid", "area"],
    )
    centroid_map = {
        int(lbl): (float(cy), float(cx))
        for lbl, cy, cx in zip(rpt["label"], rpt["centroid-0"], rpt["centroid-1"])
    }
    area_map = {int(lbl): int(a) for lbl, a in zip(rpt["label"], rpt["area"])}

    # ── adjacency & proximity metrics ────────────────────────
    adj = compute_adjacency(cell_labels, proximity_px=proximity_px)

    # ── per-cell YAP quantification ──────────────────────────
    rows = []
    for cid, nid in cell_nuc.items():
        cell_mask = cell_labels == cid
        nuc_mask  = nuc_labels  == nid
        cyto_mask = cell_mask & ~nuc_mask

        nuc_px  = yap_mip[nuc_mask].astype(np.float64) - yap_bg
        cyto_px = yap_mip[cyto_mask].astype(np.float64) - yap_bg
        if nuc_px.size == 0 or cyto_px.size == 0:
            continue

        nuc_mean  = float(np.mean(nuc_px))
        cyto_mean = float(np.mean(cyto_px))
        ratio     = nuc_mean / cyto_mean if cyto_mean > 0 else float("nan")

        cy, cx   = centroid_map.get(cid, (float("nan"), float("nan")))
        cell_area = area_map.get(cid, int(cell_mask.sum()))
        nuc_area  = int(nuc_mask.sum())
        cyto_area = int(cyto_mask.sum())

        rows.append(dict(
            cell_id                = cid,
            nucleus_id             = nid,
            yap_background_median  = round(yap_bg, 4),
            nuclear_mean_int       = round(nuc_mean,  4),
            cyto_mean_int          = round(cyto_mean, 4),
            ratio_nuc_cyto         = round(ratio, 4) if not np.isnan(ratio) else float("nan"),
            nuclear_area_px        = nuc_area,
            cell_area_px           = cell_area,
            cyto_area_px           = cyto_area,
            nuc_area_fraction      = round(nuc_area / (cell_area + 1e-9), 4),
            centroid_y             = round(cy, 1),
            centroid_x             = round(cx, 1),
        ))

    df_cells = pd.DataFrame(rows)

    # ── merge cell shape ──────────────────────────────────────
    if len(df_cells):
        # Merge cell-shape metrics by cell_id (index of cell_shape)
        df_cells = df_cells.merge(
            cell_shape,
            left_on="cell_id",
            right_index=True,
            how="left",
        )

        # Merge nuclear-shape metrics by nucleus_id (index of nuc_shape)
        df_cells = df_cells.merge(
            nuc_shape,
            left_on="nucleus_id",
            right_index=True,
            how="left",
        )

        # Merge adjacency metrics by cell_id
        df_cells = df_cells.merge(
            adj,
            on="cell_id",
            how="left",
        )

        # Round all float columns for tidy output
        for col in df_cells.select_dtypes("float64").columns:
            df_cells[col] = df_cells[col].round(4)

    # ── whole-image row (background-corrected) ────────────────
    wn  = nuc_labels  > 0
    wc  = cell_labels > 0
    wcy = wc & ~wn
    wn_mean = float(np.mean(yap_mip[wn].astype(np.float64)  - yap_bg)) if wn.any()  else float("nan")
    wc_mean = float(np.mean(yap_mip[wcy].astype(np.float64) - yap_bg)) if wcy.any() else float("nan")
    w_ratio = wn_mean / wc_mean if wc_mean > 0 else float("nan")

    whole = pd.DataFrame([dict(
        cell_id                = "WHOLE_IMAGE",
        nucleus_id             = "ALL",
        yap_background_median  = round(yap_bg, 4),
        nuclear_mean_int       = round(wn_mean, 4),
        cyto_mean_int          = round(wc_mean, 4),
        ratio_nuc_cyto         = round(w_ratio, 4) if not np.isnan(w_ratio) else float("nan"),
        nuclear_area_px        = int(wn.sum()),
        cell_area_px           = int(wc.sum()),
        cyto_area_px           = int(wcy.sum()),
    )])
    df_all = pd.concat([df_cells, whole], ignore_index=True)

    # ── ratio heat-map (vectorised index lookup) ──────────────
    ratio_lookup = np.zeros(int(cell_labels.max()) + 1, dtype=np.float32)
    for _, r in df_cells.iterrows():
        if not np.isnan(r.ratio_nuc_cyto):
            ratio_lookup[int(r.cell_id)] = float(r.ratio_nuc_cyto)
    ratio_map = ratio_lookup[cell_labels]

    return df_cells, df_all, ratio_map


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — Overview figure  (3 rows × 4 cols)
# ══════════════════════════════════════════════════════════════

def make_figure(yap, phall, hoechst,
                nuc_labels, cell_labels,
                ratio_map, df_cells,
                out_path: Path):
    """
    Row 0:  raw channels + composite
    Row 1:  Cellpose nuclear labels | cell labels | boundary overlays
    Row 2:  YAP+nuclear outlines | YAP+cell outlines | ratio heatmap | scatter
    """
    fig, axes = plt.subplots(3, 4, figsize=(22, 17),
                              gridspec_kw=dict(hspace=0.35, wspace=0.06))
    fig.patch.set_facecolor("#0d0d0d")

    yn = norm01(yap)
    pn = norm01(phall)
    hn = norm01(hoechst)

    def _show(ax, img, cmap="gray", vmin=None, vmax=None, title=""):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", aspect="equal")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")

    def _label(ax, lbl, title=""):
        from skimage.color import label2rgb
        col = label2rgb(lbl, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(col, aspect="equal", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")

    # row 0 ────────────────────────────────────────────────────
    _show(axes[0, 0], yn, "hot",      title="YAP/TAZ  (ch 0)")
    _show(axes[0, 1], pn, "Greens_r", title="Phalloidin  (ch 1)")
    _show(axes[0, 2], hn, "Blues_r",  title="Hoechst / Nucleus  (ch 2)")
    composite = np.stack([yn, pn, hn], axis=-1)
    axes[0, 3].imshow(composite, aspect="equal", interpolation="nearest")
    axes[0, 3].set_title("Composite  R:YAP | G:Phall | B:Hoechst",
                          color="white", fontsize=10, pad=4)
    axes[0, 3].axis("off")

    # row 1 ────────────────────────────────────────────────────
    _label(axes[1, 0], nuc_labels,
           f"Cellpose nuclei labels  ({nuc_labels.max()} nuclei)")
    _label(axes[1, 1], cell_labels,
           f"Cellpose cell labels  ({cell_labels.max()} cells)")

    # Compute boundaries once — reused in rows 1 and 2
    nuc_bnd  = segmentation.find_boundaries(nuc_labels,  mode="outer")
    cell_bnd = segmentation.find_boundaries(cell_labels, mode="outer")

    # ── row 1, col 2: colored cell labels + per-cell coverage % ──
    from skimage.color import label2rgb
    cell_col = label2rgb(cell_labels, bg_label=0, bg_color=(0, 0, 0))
    axes[1, 2].imshow(cell_col, aspect="equal", interpolation="nearest")
    total_px = float(cell_labels.size)
    if len(df_cells):
        for _, r in df_cells.iterrows():
            cid          = int(r.cell_id)
            coverage_pct = r.cell_area_px / total_px * 100
            axes[1, 2].text(
                r.centroid_x, r.centroid_y,
                f"#{cid}\n{coverage_pct:.1f}%",
                fontsize=5.5, color="white", ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#00000080", edgecolor="none"),
            )
    total_coverage = (cell_labels > 0).sum() / total_px * 100
    axes[1, 2].set_title(
        f"Cell labels + coverage  "
        f"({int(cell_labels.max())} cells  |  {total_coverage:.1f} % field)",
        color="white", fontsize=10, pad=4)
    axes[1, 2].axis("off")

    # ── row 1, col 3: composite + outlines (moved from col 2) ────
    comp_with_bounds = composite.copy()
    comp_with_bounds[nuc_bnd]  = [0.2, 0.9, 1.0]   # cyan = nuclei
    comp_with_bounds[cell_bnd] = [1.0, 0.85, 0.0]  # yellow = cells
    axes[1, 3].imshow(comp_with_bounds, aspect="equal", interpolation="nearest")
    axes[1, 3].set_title("Composite + outlines  (cyan=nuc | yellow=cell)",
                          color="white", fontsize=10, pad=4)
    axes[1, 3].axis("off")

    # row 2 ────────────────────────────────────────────────────
    # YAP + nuclear outlines with cell-id labels
    yap_nuc = cm.hot(yn)[:, :, :3].copy()
    yap_nuc[nuc_bnd] = [0.2, 0.9, 1.0]
    axes[2, 0].imshow(yap_nuc, aspect="equal", interpolation="nearest")
    axes[2, 0].set_title("YAP/TAZ + nuclear outlines", color="white", fontsize=10, pad=4)
    if len(df_cells):
        for _, r in df_cells.iterrows():
            axes[2, 0].text(r.centroid_x, r.centroid_y, str(int(r.cell_id)),
                            fontsize=6, color="cyan", ha="center", va="center",
                            fontweight="bold")
    axes[2, 0].axis("off")

    # YAP + cell outlines
    yap_cell = cm.hot(yn)[:, :, :3].copy()
    yap_cell[cell_bnd] = [1.0, 0.85, 0.0]
    axes[2, 1].imshow(yap_cell, aspect="equal", interpolation="nearest")
    axes[2, 1].set_title("YAP/TAZ + cell outlines", color="white", fontsize=10, pad=4)
    if len(df_cells):
        for _, r in df_cells.iterrows():
            axes[2, 1].text(r.centroid_x, r.centroid_y, str(int(r.cell_id)),
                            fontsize=6, color="yellow", ha="center", va="center",
                            fontweight="bold")
    axes[2, 1].axis("off")

    # ratio heat-map
    vmax_ratio = max(float(ratio_map.max()), 3.0)
    masked = np.ma.masked_where(cell_labels == 0, ratio_map)
    im = axes[2, 2].imshow(masked, cmap="RdBu_r", vmin=0, vmax=vmax_ratio,
                            interpolation="nearest", aspect="equal")
    axes[2, 2].set_title("YAP N/C ratio heatmap", color="white", fontsize=10, pad=4)
    axes[2, 2].axis("off")
    div = make_axes_locatable(axes[2, 2])
    cax = div.append_axes("right", size="4%", pad=0.05)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("N/C ratio", color="white", fontsize=8)
    cax.set_facecolor("#0d0d0d")

    # per-cell scatter
    ax = axes[2, 3]
    ax.set_facecolor("#1a1a1a")
    if len(df_cells):
        sc = ax.scatter(df_cells["cyto_mean_int"], df_cells["nuclear_mean_int"],
                        c=df_cells["ratio_nuc_cyto"], cmap="RdBu_r",
                        vmin=0, vmax=vmax_ratio,
                        s=70, edgecolors="white", linewidths=0.4, alpha=0.9)
        for _, r in df_cells.iterrows():
            ax.annotate(str(int(r.cell_id)),
                        (r.cyto_mean_int, r.nuclear_mean_int),
                        fontsize=6, color="white", alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
               max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "--", color="#888", lw=0.9, label="N = C")
        cb2 = fig.colorbar(sc, ax=ax, pad=0.02)
        cb2.ax.tick_params(colors="white", labelsize=8)
        cb2.set_label("N/C ratio", color="white", fontsize=8)
        ax.set_xlabel("Cytoplasmic mean intensity", color="white", fontsize=9)
        ax.set_ylabel("Nuclear mean intensity",     color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#555")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", edgecolor="#555")
    ax.set_title("Per-cell YAP/TAZ  Nuclear vs Cytoplasmic",
                 color="white", fontsize=10, pad=4)

    fig.suptitle("YAP/TAZ Nuclear/Cytoplasmic Analysis  [Cellpose]",
                 color="white", fontsize=14, y=0.998, fontweight="bold")
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Overview figure → {out_path}")


# ══════════════════════════════════════════════════════════════
#  SECTION 8 — Per-cell panel
# ══════════════════════════════════════════════════════════════

def make_cell_panels(yap, phall, hoechst,
                     nuc_labels, cell_labels,
                     df_cells, out_path: Path):
    """One row per cell: YAP crop | Phalloidin crop | Hoechst+nuc | overlay + stats"""
    n = len(df_cells)
    if n == 0:
        print("  No cells to panel.")
        return

    ncols, pad = 4, 30
    fig, axes = plt.subplots(n, ncols,
                              figsize=(ncols * 3, n * 3 + 0.5),
                              gridspec_kw=dict(hspace=0.5, wspace=0.08))
    fig.patch.set_facecolor("#0d0d0d")
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (_, row) in enumerate(df_cells.iterrows()):
        cid = int(row.cell_id)
        cm_ = (cell_labels == cid)
        props = measure.regionprops(cm_.astype(np.uint8))
        if not props:
            continue
        r0, c0, r1, c1 = props[0].bbox
        r0, c0 = max(0, r0-pad), max(0, c0-pad)
        r1, c1 = min(yap.shape[0], r1+pad), min(yap.shape[1], c1+pad)

        def crop(arr): return arr[r0:r1, c0:c1]

        yap_c   = norm01(crop(yap))
        phall_c = norm01(crop(phall))
        hoech_c = norm01(crop(hoechst))
        nuc_c   = (crop(nuc_labels) > 0).astype(np.uint8)
        cell_c  = crop(cm_).astype(np.uint8)

        # composite crop with outlines
        ov = np.stack([yap_c, phall_c, hoech_c], axis=-1)
        nb = segmentation.find_boundaries(crop(nuc_labels),  mode="outer")
        cb = segmentation.find_boundaries(crop(cell_labels), mode="outer")
        ov_b = ov.copy(); ov_b[nb] = [0.2, 0.9, 1.0]; ov_b[cb] = [1.0, 0.85, 0.0]

        r_str = (f"{row.ratio_nuc_cyto:.2f}"
                 if not np.isnan(row.ratio_nuc_cyto) else "N/A")

        def _ax(ax, img, cmap, title):
            ax.imshow(img, cmap=cmap, interpolation="nearest", aspect="equal")
            ax.set_title(title, color="white", fontsize=7.5, pad=3)
            ax.axis("off")

        _ax(axes[i, 0], yap_c,   "hot",    f"Cell {cid} | YAP/TAZ")
        _ax(axes[i, 1], phall_c, "Greens_r", f"Cell {cid} | Phalloidin")
        _ax(axes[i, 2], hoech_c, "Blues_r",  f"Cell {cid} | Hoechst")
        axes[i, 3].imshow(ov_b, aspect="equal", interpolation="nearest")
        axes[i, 3].set_title(
            f"Cell {cid}  N={row.nuclear_mean_int:.0f}  "
            f"C={row.cyto_mean_int:.0f}  R={r_str}",
            color="white", fontsize=7.5, pad=3)
        axes[i, 3].axis("off")

    fig.suptitle("Per-cell ROI  |  YAP/TAZ  [Cellpose]",
                 color="white", fontsize=12, y=1.001, fontweight="bold")
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Per-cell panel → {out_path}")


# ══════════════════════════════════════════════════════════════
#  SECTION 9 — Orchestration
# ══════════════════════════════════════════════════════════════

def process_file(filepath: str, out_dir: Path, args) -> pd.DataFrame:
    stem = Path(filepath).stem
    print(f"\n{'='*60}\n  {filepath}\n{'='*60}")

    mip = load_czi(filepath)
    for ch_name, ch_idx in [("YAP", args.ch_yap),
                              ("Phalloidin", args.ch_phall),
                              ("Hoechst", args.ch_nuc)]:
        if ch_idx >= mip.shape[0]:
            raise ValueError(f"Channel {ch_idx} ({ch_name}) out of range "
                             f"— image has only {mip.shape[0]} channels.")

    yap     = mip[args.ch_yap]
    phall   = mip[args.ch_phall]
    hoechst = mip[args.ch_nuc]
    print(f"  Image size: {yap.shape[0]} × {yap.shape[1]} px")

    # ── segmentation ──────────────────────────────────────────
    nuc_labels  = make_nuclear_mask(
        hoechst, diameter=args.diameter,
        gpu=args.gpu, min_size=args.min_nuc_px,
        batch_size=args.batch_size,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.nuc_cellprob,
        augment=args.augment,
        tile_overlap=args.tile_overlap,
    )
    cell_labels = make_cell_mask(
        phall, hoechst, diameter=args.diameter,
        gpu=args.gpu, min_size=args.min_cell_px,
        batch_size=args.batch_size,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cell_cellprob,
        augment=args.augment,
        tile_overlap=args.tile_overlap,
    )

    # ── quantify ──────────────────────────────────────────────
    df_cells, df_all, ratio_map = quantify_yap(
        yap, nuc_labels, cell_labels,
        proximity_px=args.proximity_px,
    )
    print(f"  Cells with paired nuclei: {len(df_cells)}")

    # ── save CSV ──────────────────────────────────────────────
    csv_path = out_dir / f"{stem}_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"  CSV → {csv_path}")

    # ── figures ───────────────────────────────────────────────
    make_figure(yap, phall, hoechst,
                nuc_labels, cell_labels,
                ratio_map, df_cells,
                out_dir / f"{stem}_overview.png")

    make_cell_panels(yap, phall, hoechst,
                     nuc_labels, cell_labels, df_cells,
                     out_dir / f"{stem}_cell_panels.png")

    return df_all


# ══════════════════════════════════════════════════════════════
#  SECTION 10 — CLI
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="YAP/TAZ N/C analysis — Cellpose segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",       help=".czi file or directory of .czi files")
    p.add_argument("-o","--output", default="yap_taz",
                   help="Output directory (default: yap_taz)")
    p.add_argument("--ch-yap",   type=int, default=0, help="YAP/TAZ channel (default 0)")
    p.add_argument("--ch-phall", type=int, default=1, help="Phalloidin channel (default 1)")
    p.add_argument("--ch-nuc",   type=int, default=2, help="Hoechst channel (default 2)")
    p.add_argument("--diameter",   type=float, default=None,
                   help="Expected cell/nucleus diameter in pixels (None = auto-estimate)")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Cellpose tiles per forward pass. "
                        "Defaults to 8 (CPU) or 32 (GPU). "
                        "Raise on high-VRAM cards (64, 128) for more throughput.")
    p.add_argument("--min-nuc-px",  type=int, default=200,
                   help="Min nucleus size in px (default 200)")
    p.add_argument("--min-cell-px", type=int, default=500,
                   help="Min cell size in px (default 500)")

    # ── Cellpose sensitivity (C2C12 myoblast-tuned defaults) ──
    p.add_argument("--flow-threshold", type=float, default=0.6,
                   help="Cellpose flow threshold — raise (0.6–0.9) to recover elongated "
                        "cells; lower (0.4) for round cells  (default 0.6)")
    p.add_argument("--nuc-cellprob", type=float, default=0.0,
                   help="Cell-probability threshold for NUCLEI  (default 0.0; "
                        "lower to -2 … -6 to catch dim nuclei)")
    p.add_argument("--cell-cellprob", type=float, default=-1.0,
                   help="Cell-probability threshold for CELLS  (default -1.0; "
                        "lower to -4 … -6 to catch faint / compressed cells)")
    p.add_argument("--augment", action="store_true", default=True,
                   help="4-fold rotation augmentation — better for elongated cells "
                        "(on by default; --no-augment to disable)")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--tile-overlap", type=float, default=0.3,
                   help="Tile overlap fraction  (default 0.3 — high value prevents "
                        "elongated cells from being clipped at tile edges)")

    # ── adjacency / proximity ─────────────────────────────────
    p.add_argument("--proximity-px", type=int, default=10,
                   help="Pixel radius for 'nearby cell' metric  (default 10)")
    args = p.parse_args()

    # ── GPU verification ──────────────────────────────────────
    if args.gpu:
        gpu_ok = check_gpu()
        if not gpu_ok:
            print("  [WARNING] --gpu requested but no usable GPU found.")
            print("            Continuing on CPU — remove --gpu to silence this.\n")
            args.gpu = False   # prevent Cellpose from raising an error

    # ── auto batch_size: 8 on CPU, 32 on GPU (user can override) ─
    if args.batch_size is None:
        args.batch_size = 32 if args.gpu else 8
    print(f"  batch_size = {args.batch_size}  ({'GPU' if args.gpu else 'CPU'})")

    inp     = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files   = sorted(inp.glob("*.czi")) if inp.is_dir() else [inp]

    if not files:
        print(f"No .czi files found in {inp}"); sys.exit(1)

    all_dfs = []
    for f in files:
        try:
            df = process_file(str(f), out_dir, args)
            df.insert(0, "file", f.name)
            all_dfs.append(df)
        except Exception as e:
            print(f"  !! Error: {f.name}: {e}", file=sys.stderr)

    if len(all_dfs) > 1:
        comb = pd.concat(all_dfs, ignore_index=True)
        cp   = out_dir / "ALL_FILES_combined_results.csv"
        comb.to_csv(cp, index=False)
        print(f"\nCombined CSV → {cp}")

    print(f"\n✓  Results in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()