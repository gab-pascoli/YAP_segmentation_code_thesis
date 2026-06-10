# YAP/TAZ Nuclear–Cytoplasmic Intensity Analyser
Code used for cell segmentation and YAP/TAZ nucleocytoplasmic assesment during my thesis titled: "Influence of hypergravity on mechanotransduction of C2C12 myoblasts"

Automated quantification of YAP/TAZ subcellular localisation from multi-channel
fluorescence microscopy images (`.czi` format).  The pipeline segments every
cell and nucleus using the [Cellpose](https://github.com/MouseLand/cellpose)
deep-learning model, measures background-corrected YAP/TAZ intensity in each
compartment, and reports the nuclear-to-cytoplasmic (N/C) intensity ratio — the
standard readout of YAP/TAZ transcriptional activity — alongside cell shape and
cell-contact metrics.

---

## Table of Contents

1. [Biological Background](#1-biological-background)
2. [Imaging Requirements](#2-imaging-requirements)
3. [Dependencies and Installation](#3-dependencies-and-installation)
4. [Quick Start](#4-quick-start)
5. [Full Command-Line Reference](#5-full-command-line-reference)
6. [How the Pipeline Works](#6-how-the-pipeline-works)
7. [Output Files](#7-output-files)
8. [CSV Column Reference](#8-csv-column-reference)
9. [Adapting to Other Cell Types](#9-adapting-to-other-cell-types)
10. [GPU Acceleration](#10-gpu-acceleration)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Biological Background

**YAP** (Yes-Associated Protein) and **TAZ** (Transcriptional Co-Activator with
PDZ-binding motif) are the terminal effectors of the Hippo signalling pathway.
Their activity is governed primarily by subcellular localisation rather than
expression level:

- **Nuclear YAP/TAZ** is transcriptionally active.  It binds TEAD family
  transcription factors and drives expression of pro-proliferative and
  mechanosensitive target genes including *CTGF* and *CYR61*.
- **Cytoplasmic YAP/TAZ** is inactive.  The Hippo pathway, triggered by
  high cell density, cell-cell contact, and soft substrates, causes YAP/TAZ
  phosphorylation, cytoplasmic sequestration, and proteasomal degradation.

Because total YAP/TAZ expression can vary between cells, the **nuclear-to-cytoplasmic
(N/C) intensity ratio** — rather than absolute nuclear intensity — is the correct
measure of activity.  A ratio greater than 1 indicates nuclear enrichment (active);
a ratio below 1 indicates cytoplasmic retention (inactive).

This pipeline measures the N/C ratio per cell from antibody-stained fluorescence
images and simultaneously records cell shape and contact state, enabling
correlation of YAP/TAZ localisation with morphological and mechanical context.

---

## 2. Imaging Requirements

### File format
Zeiss `.czi` files (single file or directory of files).  Both Z-stack and
single-plane acquisitions are supported.  Z-stacks are automatically collapsed
to a maximum-intensity projection (see [Section 6.1](#61-loading-and-mip-projection)).

### Required channels (0-indexed)
| Default index | Stain | Purpose |
|:---:|---|---|
| 0 | YAP/TAZ antibody | Signal to be quantified |
| 1 | Phalloidin (F-actin) | Defines whole-cell boundary |
| 2 | Hoechst (DNA) | Defines nuclear boundary |

If your channel order differs, use `--ch-yap`, `--ch-phall`, and `--ch-nuc` to
reassign (see [Section 5](#5-full-command-line-reference)).

### Staining quality
- The Hoechst signal should be bright and compact to allow accurate nuclear
  segmentation.  Weak or diffuse Hoechst staining will reduce nuclear detection
  sensitivity.
- The Phalloidin signal should outline cell bodies clearly.  Very faint
  Phalloidin staining can cause under-segmentation of cells; lower
  `--cell-cellprob` in that case (see [Section 11](#11-troubleshooting)).

---

## 3. Dependencies and Installation

### Python version
Python 3.10 or later is required (the code uses the `X | Y` union type hint
syntax introduced in 3.10).

### Install dependencies

```bash
pip install cellpose bioio bioio-czi czifile \
            numpy pandas matplotlib scipy scikit-image
```

If `bioio-czi` is unavailable or fails, the pipeline falls back automatically
to `czifile`, which handles most older `.czi` files.

### First-run model download
On the first execution, Cellpose automatically downloads the `nuclei` and `cyto3`
model weights (~250 MB total) to `~/.cellpose/models/`.  An internet connection
is required only for this initial download; all subsequent runs are fully offline.

### GPU support (optional)
GPU acceleration is not required but significantly speeds up segmentation of
large images or batch processing.  See [Section 10](#10-gpu-acceleration) for
setup instructions.

---

## 4. Quick Start

### Single image
```bash
python yap_taz_cellpose.py image.czi
```
Results are written to a new `yap_taz/` directory in the current folder.

### Single image with a custom output directory
```bash
python yap_taz_cellpose.py image.czi -o results/experiment_01/
```

### Batch processing — all `.czi` files in a folder
```bash
python yap_taz_cellpose.py /path/to/images/ -o results/
```
Each file is processed independently.  A combined CSV across all files is
written at the end as `ALL_FILES_combined_results.csv`.

### With GPU and a known cell diameter
```bash
python yap_taz_cellpose.py image.czi --gpu --diameter 40
```

---

## 5. Full Command-Line Reference

```
python yap_taz_cellpose.py <input> [options]
```

### Positional argument

| Argument | Description |
|---|---|
| `input` | Path to a `.czi` file, or a directory containing `.czi` files |

### General options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | `yap_taz` | Directory to write all output files |
| `--ch-yap` | `0` | Channel index for the YAP/TAZ antibody stain |
| `--ch-phall` | `1` | Channel index for Phalloidin (F-actin / cell body) |
| `--ch-nuc` | `2` | Channel index for Hoechst (nucleus) |
| `--diameter` | auto | Expected cell or nucleus diameter in pixels.  `None` = Cellpose auto-estimates from the image.  Providing the correct value improves segmentation accuracy, especially for very small or very large cells |
| `--gpu` | off | Enable GPU acceleration via CUDA.  The pipeline verifies GPU availability before starting and falls back to CPU if none is found |
| `--batch-size` | 8 (CPU) / 32 (GPU) | Number of image tiles processed per Cellpose forward pass.  Increase to 64 or 128 on high-VRAM GPUs for faster throughput |

### Size filters

| Flag | Default | Description |
|---|---|---|
| `--min-nuc-px` | `200` | Minimum nucleus area in pixels.  Objects below this threshold are discarded as debris or segmentation artefacts |
| `--min-cell-px` | `500` | Minimum cell area in pixels.  Fragments smaller than this are discarded |

### Cellpose sensitivity — tuned for C2C12 myoblasts

These parameters control how strictly Cellpose accepts candidate cell and nucleus
regions.  The defaults are optimised for **elongated cells** (e.g. C2C12 myoblasts).
See [Section 9](#9-adapting-to-other-cell-types) for round-cell recommendations.

| Flag | Default | Description |
|---|---|---|
| `--flow-threshold` | `0.6` | Minimum flow-field consistency score for a region to be accepted as a cell.  **Raise** (0.6–0.9) to recover elongated or irregularly shaped cells.  **Lower** (0.4) for compact, round cells |
| `--nuc-cellprob` | `0.0` | Minimum predicted cell-probability for nuclear pixels.  **Lower** (−2 to −6) to detect dim or weakly stained nuclei.  **Raise** (up to 2) to reduce false nuclear detections |
| `--cell-cellprob` | `-2.0` | Minimum predicted cell-probability for cell pixels.  The negative default accepts low-probability regions at cell edges and in dense clusters.  **Lower** further (−4 to −6) for faint or compressed cells |
| `--augment` / `--no-augment` | on | 4-fold rotation test-time augmentation.  The image is segmented at 0°, 90°, 180°, and 270° and results are merged.  This substantially improves accuracy for elongated cells but roughly quadruples runtime.  Disable with `--no-augment` if speed is critical |
| `--tile-overlap` | `0.3` | Fractional overlap between adjacent image tiles used during Cellpose inference.  A high value (0.3) prevents elongated cells from being split at tile boundaries.  Reduce to 0.1 for round cells to speed up processing |

### Cell contact and proximity

| Flag | Default | Description |
|---|---|---|
| `--proximity-px` | `10` | Pixel dilation radius used to define a "proximity zone" around each cell.  Cells in a different label whose pixels fall within this zone (but do not touch) are counted as `n_nearby` in the output |

---

## 6. How the Pipeline Works

The pipeline executes the following steps in order for each image.

### 6.1 Loading and MIP projection

The `.czi` file is read using `bioio-czi` (preferred) or `czifile` (fallback).
Both produce a 4-D array of shape `(channels, Z, Y, X)`.

A **maximum-intensity projection (MIP)** is then computed along the Z axis,
producing a 2-D image `(Y, X)` for each channel.  The MIP keeps, for every
pixel position (x, y), the brightest value observed across all focal planes.
This ensures that signal from structures that are not all in the same focal plane —
for example nuclei at slightly different depths in a cell monolayer — is
retained in full.  Single-plane acquisitions (no Z-stack) pass through unchanged.

### 6.2 Nuclear segmentation — Hoechst channel

The Cellpose **`nuclei`** deep-learning model is applied to the Hoechst MIP.
The model was trained on tens of thousands of nucleus images and can detect
nuclei of a wide range of sizes and staining intensities.

Internally, Cellpose predicts two things for every pixel:
1. A **cell-probability score** — how likely is this pixel to belong to a nucleus.
2. A **flow field** — a vector pointing toward the nearest nucleus centre.

Pixels above `--nuc-cellprob` whose flow field is consistent with a nucleus
interior (flow consistency score above `--flow-threshold`) are accepted.
Objects smaller than `--min-nuc-px` pixels are discarded.

The output is an **integer label image** of the same size as the Hoechst channel:
every pixel belonging to nucleus *N* has the value *N*, and background pixels
have the value 0.  Each positive integer therefore uniquely identifies one nucleus.

If Cellpose cannot be loaded (e.g. not installed), the function falls back to
classical **Otsu thresholding + distance-transform watershed**: Otsu finds the
intensity threshold that best separates foreground from background, binary fill
and small-object removal clean the mask, and watershed splits touching nuclei by
flooding from local intensity peaks (object centres) outward.

### 6.3 Cell segmentation — Phalloidin + Hoechst channels

The Cellpose **`cyto3`** model is applied to a two-channel input: the Phalloidin
MIP as the cytoplasm signal and the Hoechst MIP as an optional nuclear guide.
Providing both channels helps the model resolve the boundary between cells that
are in close contact, because the nucleus positions provide additional anchoring
information.

The same flow-field and probability-threshold logic as nuclear segmentation
applies, but with defaults relaxed for elongated cells (see [Section 5](#5-full-command-line-reference)).
4-fold rotation augmentation is **on by default** for cell segmentation because
it substantially improves detection of spindle-shaped myoblasts.

The output is an integer label image where each positive value uniquely identifies
one cell body (including the nucleus region inside it).

The same Otsu + watershed fallback applies if Cellpose is unavailable, using
the Phalloidin channel alone.

### 6.4 Pairing cells with nuclei

Cellpose segments cells and nuclei independently using separate label spaces: the
number 3 in the cell label map has no inherent relationship with the number 3 in
the nuclear label map.  This step establishes the correspondence.

For every segmented cell, the algorithm finds the nucleus with the **greatest
pixel overlap** — the nucleus whose pixels occupy the most area inside that cell.
Cells that do not overlap any nuclear pixels (e.g. cytoplasm fragments without a
nucleus, or cells at the very edge of the image that were only partially captured)
are excluded from all downstream analysis.

This matching is performed efficiently with a single vectorised operation:
all overlapping (cell, nucleus) pixel pairs are encoded as integers and counted
with `numpy.bincount`, producing an overlap count matrix in one call rather than
looping over individual cells.

### 6.5 Background correction

Before any intensity measurement, a **background correction** is applied.

The median YAP/TAZ intensity of all pixels that do not belong to any cell
(`cell_labels == 0`) is computed and subtracted from every nuclear and
cytoplasmic pixel value.  This removes the combined contribution of camera
read noise, dark current, and cellular autofluorescence — none of which
represents genuine YAP/TAZ antibody signal.

In very dense fields where no background pixels exist, the correction is skipped
(offset set to zero) and a warning is printed.  The background value applied is
recorded in every output row as `yap_background_median` for traceability.

### 6.6 YAP/TAZ intensity quantification

For each matched cell-nucleus pair:

1. **Nuclear mask** — all pixels belonging to the matched nucleus label.
2. **Cytoplasm mask** — all pixels inside the cell boundary that are *not* in
   the nucleus (cell mask minus nucleus mask).
3. **Background-corrected intensities** — the raw pixel values from the YAP/TAZ
   channel are extracted for each compartment and the background median is subtracted.
4. **Compartment means** — the mean of the corrected nuclear pixels and the mean
   of the corrected cytoplasmic pixels are computed separately.
5. **N/C ratio** — `nuclear_mean / cytoplasmic_mean`.  If the cytoplasmic mean
   is zero or negative after background correction (very rare; can occur in
   extremely sparse cells or near-background signal), the ratio is recorded as
   `NaN`.

A **WHOLE_IMAGE** summary row is appended to the CSV, aggregating all nuclear
pixels and all cytoplasmic pixels across the entire field into a single
field-level N/C ratio.

### 6.7 Cell and nuclear shape metrics

For every cell and nucleus, a set of morphological descriptors is computed from
the segmentation masks using `skimage.measure.regionprops_table`.  These are
appended to the per-cell results row so that YAP/TAZ localisation can be
correlated with cell morphology in downstream statistical analysis.

Shape metrics are reported twice: once for the **cell** (unprefixed columns)
and once for its matched **nucleus** (columns prefixed `nuc_`).

See [Section 8](#8-csv-column-reference) for the full list and definitions.

### 6.8 Cell adjacency and proximity metrics

Cell-cell contact suppresses YAP/TAZ nuclear localisation through contact
inhibition; isolated cells tend to show the opposite.  This step quantifies the
contact and crowding state of each cell.

For each cell, the pipeline computes:
- **Direct contact** — how many boundary pixels of this cell are directly adjacent
  to a pixel belonging to a different cell (`contact_px`, `contact_frac`), and
  how many other cells are involved (`n_touching`).
- **Proximity** — the cell mask is dilated outward by `--proximity-px` pixels to
  create a zone of influence.  Non-touching cells whose pixels fall within this
  zone are counted as `n_nearby`.
- **Nearest-neighbour distance** — Euclidean distance between this cell's centroid
  and the nearest other cell's centroid (`nearest_dist_px`).
- **Clustering flag** — `is_clustered = 1` if the cell touches or is near at
  least one other cell; `0` if fully isolated.

Steps 1–4 are fully vectorised using NumPy operations on the full label image.
The proximity dilation (step 5) uses a Python loop over cells; this is fast
enough for typical field sizes (fewer than ~500 cells per image).

### 6.9 Figure generation

Two figures are saved for every processed image.

**Overview figure** (`_overview.png`, 3 rows × 4 columns):

| Row | Column 0 | Column 1 | Column 2 | Column 3 |
|:---:|---|---|---|---|
| 0 | YAP/TAZ channel | Phalloidin channel | Hoechst channel | RGB composite |
| 1 | Cellpose nuclear labels | Cellpose cell labels | Cell labels + coverage % per cell | Composite + outlines (cyan = nucleus, yellow = cell) |
| 2 | YAP + nuclear outlines + cell IDs | YAP + cell outlines + cell IDs | N/C ratio heat-map | Scatter: cytoplasmic vs nuclear mean, coloured by N/C ratio |

**Per-cell panel** (`_cell_panels.png`, one row per cell):

Each row shows a zoomed crop (bounding box + 30 px padding) of one cell across
four panels: YAP/TAZ, Phalloidin, Hoechst, and a composite overlay with
nucleus (cyan) and cell (yellow) outlines.  The title of the overlay panel
shows the cell ID, nuclear mean intensity, cytoplasmic mean intensity, and N/C ratio.
This panel is intended for visual quality control of individual cell segmentations.

---

## 7. Output Files

For an input file named `experiment.czi` and output directory `results/`:

| File | Description |
|---|---|
| `results/experiment_results.csv` | Per-cell results table (see [Section 8](#8-csv-column-reference)) |
| `results/experiment_overview.png` | Full-field overview figure (3 × 4 panels) |
| `results/experiment_cell_panels.png` | Zoomed per-cell gallery |
| `results/ALL_FILES_combined_results.csv` | All per-cell rows from all processed images combined into a single CSV (batch mode only; includes a `file` column identifying the source image) |

---

## 8. CSV Column Reference

Each row in the results CSV corresponds to one segmented cell (except the final
`WHOLE_IMAGE` summary row).

### Core intensity columns

| Column | Description |
|---|---|
| `file` | Source filename (batch mode only) |
| `cell_id` | Integer label assigned to this cell by Cellpose.  `WHOLE_IMAGE` in the summary row |
| `nucleus_id` | Integer label of the matched nucleus.  `ALL` in the summary row |
| `yap_background_median` | Median YAP/TAZ intensity of all background pixels (outside all cell masks); subtracted from nuclear and cytoplasmic means |
| `nuclear_mean_int` | Background-corrected mean YAP/TAZ pixel intensity inside the nucleus |
| `cyto_mean_int` | Background-corrected mean YAP/TAZ pixel intensity in the cytoplasm |
| `ratio_nuc_cyto` | **N/C ratio** = `nuclear_mean_int / cyto_mean_int`.  The primary readout.  Values > 1 indicate nuclear enrichment (active YAP/TAZ); values < 1 indicate cytoplasmic retention (inactive) |
| `nuclear_area_px` | Number of pixels in the nuclear mask |
| `cell_area_px` | Number of pixels in the whole-cell mask |
| `cyto_area_px` | Number of pixels in the cytoplasm mask (= cell − nucleus) |
| `nuc_area_fraction` | `nuclear_area_px / cell_area_px` — how much of the cell area is occupied by the nucleus |
| `centroid_y` | Y coordinate of the cell centroid in pixels |
| `centroid_x` | X coordinate of the cell centroid in pixels |

### Cell shape columns

These columns describe the morphology of the **whole cell** (including the nucleus).

| Column | Description |
|---|---|
| `area_px` | Cell area in pixels |
| `eccentricity` | 0 = perfect circle; approaching 1 = line-like.  Key descriptor for elongated myoblasts |
| `aspect_ratio` | Major axis length / minor axis length |
| `circularity` | 4π × area / perimeter².  1 = perfect circle; lower values indicate irregular or branched shapes |
| `solidity` | Area / convex hull area.  1 = convex; values < 1 indicate concave indentations |
| `extent` | Area / bounding box area |
| `major_axis_px` | Length of the major (long) axis in pixels |
| `minor_axis_px` | Length of the minor (short) axis in pixels |
| `orientation_deg` | Angle of the major axis relative to horizontal, in degrees (range −90 to +90) |

### Nuclear shape columns

Identical metrics computed for the **nucleus** only.  Column names are prefixed `nuc_`.

`nuc_area_px`, `nuc_eccentricity`, `nuc_aspect_ratio`, `nuc_circularity`,
`nuc_solidity`, `nuc_extent`, `nuc_major_axis_px`, `nuc_minor_axis_px`,
`nuc_orientation_deg`

### Cell adjacency and proximity columns

| Column | Description |
|---|---|
| `n_touching` | Number of other cells that share at least one boundary pixel with this cell |
| `contact_px` | Number of boundary pixels of this cell that directly abut another cell |
| `contact_frac` | `contact_px / perimeter`.  0 = fully isolated; 1 = completely surrounded |
| `n_nearby` | Number of non-touching cells that have a pixel within `--proximity-px` pixels |
| `nearest_dist_px` | Euclidean centroid-to-centroid distance to the nearest other cell, in pixels |
| `is_clustered` | `1` if `n_touching > 0` or `n_nearby > 0`; `0` if fully isolated |

---

## 9. Adapting to Other Cell Types

The default parameters are optimised for **C2C12 myoblasts**, which are elongated
and spindle-shaped.  Other cell types require adjustments.

### Round or epithelial cells (e.g. HeLa, MCF7, Caco-2)

```bash
python yap_taz_cellpose.py image.czi \
    --flow-threshold 0.4 \
    --cell-cellprob 0.0 \
    --no-augment \
    --tile-overlap 0.1
```

Rationale: round cells produce consistent, high-magnitude flow fields that are
well-captured at the stricter `flow-threshold=0.4` default.  Augmentation and
high tile overlap are unnecessary and slow for round cells at normal density.

### Sparse or isolated cells

If cells are widely separated, the default `--cell-cellprob -2.0` is
conservative enough.  If cells are still being missed, lower it further:

```bash
python yap_taz_cellpose.py image.czi --cell-cellprob -4.0
```

### Very dense or confluent monolayers

High-confluence images are challenging because cell boundaries are ambiguous.
Increase tile overlap and augmentation to maximise detection:

```bash
python yap_taz_cellpose.py image.czi \
    --flow-threshold 0.7 \
    --cell-cellprob -3.0 \
    --augment \
    --tile-overlap 0.4
```

### Dim Hoechst staining

If nuclear detection is poor due to weak Hoechst signal:

```bash
python yap_taz_cellpose.py image.czi --nuc-cellprob -4.0
```

### Non-standard cell sizes

If Cellpose auto-diameter estimation is unreliable (common with very small or
very large cells), provide an explicit diameter in pixels.  Measure a typical
cell or nucleus diameter in ImageJ/FIJI and pass it:

```bash
python yap_taz_cellpose.py image.czi --diameter 25   # small cells
python yap_taz_cellpose.py image.czi --diameter 80   # large cells
```

---

## 10. GPU Acceleration

GPU acceleration uses CUDA and requires an NVIDIA GPU with an appropriate driver.

### Install PyTorch with CUDA support

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the appropriate CUDA version for your driver
(`cu118` for CUDA 11.8, `cu121` for CUDA 12.1, etc.).  NVIDIA driver version
≥ 525 is required for CUDA 12.

### Run with GPU

```bash
python yap_taz_cellpose.py image.czi --gpu
```

The pipeline automatically verifies GPU availability before starting.  If a
usable GPU is not found, it prints a warning and continues on CPU; it does not
abort.  To diagnose GPU issues independently:

```python
from utils import check_gpu
check_gpu()
```

This prints the PyTorch version, CUDA build version, GPU device properties
(name, VRAM, compute capability), and Cellpose's own GPU status.

### Recommended batch sizes by hardware

| Hardware | `--batch-size` |
|---|---|
| CPU | 8 (default) |
| GPU, 4–8 GB VRAM | 32 (default) |
| GPU, 12–16 GB VRAM | 64 |
| GPU, 24+ GB VRAM | 128 |

---

## 11. Troubleshooting

### Too few cells or nuclei detected

- **Lower** `--cell-cellprob` (e.g. `--cell-cellprob -4.0`) to accept
  lower-probability cell regions.
- **Lower** `--nuc-cellprob` (e.g. `--nuc-cellprob -3.0`) for dim nuclei.
- **Raise** `--flow-threshold` (e.g. `0.7–0.9`) if elongated cells are being
  rejected.
- Provide `--diameter` explicitly if the auto-estimate is unreliable.

### Too many false positives (debris segmented as cells)

- **Raise** `--min-cell-px` to discard smaller fragments.
- **Raise** `--cell-cellprob` toward 0 or above to require higher-confidence
  cell detections.
- **Lower** `--flow-threshold` toward 0.4 to require more circular flow patterns.

### Cells split at image tile boundaries

- **Raise** `--tile-overlap` (e.g. `0.4` or `0.5`) so elongated cells are not
  truncated between adjacent tiles.

### N/C ratio is NaN for some cells

The ratio is undefined when the cytoplasmic mean is zero or negative after
background correction.  This can happen when:
- The cytoplasm mask is very thin (nucleus nearly fills the cell) — check
  `cyto_area_px` in the CSV.
- The cell is at the image edge and the cytoplasm mask is partially outside the
  frame.
- Background correction over-subtracts in very sparse images.

### `bioio-czi` fails on loading

The pipeline falls back automatically to `czifile`.  If both fail, ensure the
`.czi` file is not corrupted and that at least one of the two libraries is
installed:

```bash
pip install bioio bioio-czi     # preferred
pip install czifile              # fallback
```

### Memory errors on large images

- Reduce `--batch-size` (e.g. `--batch-size 4` on CPU).
- Ensure no other processes are consuming GPU memory if using `--gpu`.

### Cellpose model weights not downloading

Cellpose requires internet access on the first run to download model weights to
`~/.cellpose/models/`.  If the download fails (firewall, no internet on compute
node), download the weights manually on a connected machine and copy the
`~/.cellpose/` directory to the target system.

---

## Appendix: Figure Panel Reference

### Overview figure (`_overview.png`)

```
┌────────────────┬────────────────┬────────────────┬────────────────┐
│ YAP/TAZ raw    │ Phalloidin raw │ Hoechst raw    │ RGB composite  │
│ (hot cmap)     │ (green cmap)   │ (blue cmap)    │ R=YAP G=Ph B=H │
├────────────────┼────────────────┼────────────────┼────────────────┤
│ Nuclear labels │ Cell labels    │ Cell labels +  │ Composite +    │
│ (pseudo-colour)│ (pseudo-colour)│ ID + coverage% │ outlines       │
│                │                │                │ cyan=nuc       │
│                │                │                │ yellow=cell    │
├────────────────┼────────────────┼────────────────┼────────────────┤
│ YAP + nuclear  │ YAP + cell     │ N/C ratio      │ Scatter plot   │
│ outlines + IDs │ outlines + IDs │ heat-map       │ cyto vs nuc    │
│                │                │ (RdBu_r)       │ mean intensity │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

The scatter plot (bottom right) shows one point per cell.  Points above the
dashed diagonal have nuclear intensity > cytoplasmic intensity (N/C > 1, active
YAP/TAZ); points below have the inverse.  Points are coloured by N/C ratio using
the same RdBu_r colour scale as the heat-map.

### Per-cell panel (`_cell_panels.png`)

```
┌────────────────┬────────────────┬────────────────┬────────────────────────────┐
│ Cell N         │ Cell N         │ Cell N         │ Cell N  N=<nuc>  C=<cyto>  │
│ YAP/TAZ        │ Phalloidin     │ Hoechst        │ R=<ratio>                  │
│ (hot cmap)     │ (green cmap)   │ (blue cmap)    │ composite + outlines       │
├ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
│  ... one row per cell ...                                                      │
└────────────────┴────────────────┴────────────────┴────────────────────────────┘
```

Each row is cropped to the cell's bounding box plus 30 pixels of context padding.
