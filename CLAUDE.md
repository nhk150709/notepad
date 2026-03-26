# CLAUDE.md — Agent Guide

## Model & Runtime

- **Local model**: qwen3.5 (via Ollama or compatible local server)
- **Hallucination mitigation**: qwen3.5 can hallucinate on large tasks. Keep each agent invocation small and scoped.

## Agent Task Rules

1. **One module per task** — each agent call should create or edit at most one Python file.
2. **Read before edit** — always read a file before modifying it.
3. **Declare interface first** — before writing a function, state its inputs and outputs in a comment.
4. **No invented APIs** — only call functions that are listed in the Module Registry below.
5. **Update this file** — after creating or changing a module, update the Module Registry section.
6. **No large rewrites** — if a change affects more than ~50 lines, split it into multiple tasks.
7. **Verify outputs** — after writing a module, state what it returns and what files it writes.

---

## Project Overview

Iterative inverse design of a grayscale copper-density image using convolution.

- **Goal**: modify only allowed rectangular regions so that, after convolution with a Gaussian kernel, the gray values at specified monitor points converge to GV = 20 (tolerance ±10%, i.e. 18–22).
- **Image**: ~5000×5000 px, 8bpp grayscale. Each pixel = 0.1 mm.
- **GV meaning**: 255 = 100% Cu density, 0 = 0% Cu density.
- **Constraint**: pixels outside editable rectangles must not change.

---

## Input Files

| File | Description | Key Columns |
|---|---|---|
| `l1-2post_panel_nhk_modified_1_trimmed.png` | Base 8bpp grayscale image | — |
| `sections.csv` | Editable rectangular regions | `Section, Area, Average GV, Min, Max, BX, BY, Width, Height` |
| `points.csv` | Monitor points to optimize | `X, Y` |

---

## Module Architecture

Planned breakdown — one file per concern:

| Module | Purpose |
|---|---|
| `kernel.py` | Build Gaussian kernels; run FFT convolution on CPU and GPU |
| `data_io.py` | Load/save image, CSV files, and output artefacts |
| `mask.py` | Build editable pixel masks and block/tile structures from sections |
| `forward.py` | Run full forward pass: convolve image, sample at monitor points |
| `optimizer.py` | Iterative update loop (coarse-to-fine or block-based) |
| `logger.py` | Console logging per iteration and tqdm progress bars |
| `outputs.py` | Save per-iteration images, CSVs, plots, and diff images |
| `main.py` | CLI entry point; argument parsing; orchestrate all modules |

---

## Module Registry

Update this section every time a module is created or its interface changes.

---

### `kernel.py`

- **Status**: not yet implemented
- **Purpose**: generate convolution kernels and apply FFT-based convolution
- **Inputs**:
  - `sigma` (float) — kernel width
  - `kind` (str) — kernel type, e.g. `"gaussian"`
  - `trunc` (float) — truncation factor
  - `image` (ndarray H×W float32) — image to convolve
  - `use_gpu` (bool) — whether to use CuPy
- **Outputs**:
  - `make_kernel(sigma, kind, trunc) -> ndarray` — 2D kernel
  - `convolve(image, kernel, use_gpu) -> ndarray` — convolved image, same shape
- **Key functions**: `make_kernel`, `convolve`
- **Side effects**: prints backend used (CPU/GPU)

---

### `data_io.py`

- **Status**: not yet implemented
- **Purpose**: load and save all data files
- **Inputs**:
  - `image_path` (str/Path) — path to PNG
  - `sections_path` (str/Path) — path to sections.csv
  - `points_path` (str/Path) — path to points.csv
- **Outputs**:
  - `load_image(path) -> ndarray H×W uint8`
  - `load_sections(path) -> DataFrame` — columns: Section, BX, BY, Width, Height
  - `load_points(path) -> ndarray N×2 int` — [[X, Y], ...]
  - `save_image(array, path)` — writes PNG
  - `save_csv(df, path)` — writes CSV
- **Key functions**: `load_image`, `load_sections`, `load_points`, `save_image`, `save_csv`

---

### `mask.py`

- **Status**: not yet implemented
- **Purpose**: build pixel-level editable mask and block/tile structures
- **Inputs**:
  - `sections` (DataFrame) — from `data_io.load_sections`
  - `image_shape` (tuple H, W)
  - `block_size` (int) — tile size for coarse-to-fine
- **Outputs**:
  - `build_mask(sections, image_shape) -> ndarray H×W bool` — True where editable
  - `build_blocks(sections, image_shape, block_size) -> list[Block]` — list of block descriptors
- **Key functions**: `build_mask`, `build_blocks`

---

### `forward.py`

- **Status**: not yet implemented
- **Purpose**: compute convolved output and sample monitor point values
- **Inputs**:
  - `image` (ndarray H×W float32)
  - `kernel` (ndarray) — from `kernel.make_kernel`
  - `points` (ndarray N×2) — monitor point coords
  - `use_gpu` (bool)
- **Outputs**:
  - `run_forward(image, kernel, points, use_gpu) -> (conv_map, monitor_gv)`
    - `conv_map`: ndarray H×W float32 — full convolved image
    - `monitor_gv`: ndarray N float32 — GV at each monitor point
- **Key functions**: `run_forward`

---

### `optimizer.py`

- **Status**: not yet implemented
- **Purpose**: iterative update loop to drive monitor GVs toward target
- **Inputs**:
  - `image` (ndarray H×W float32) — initial image
  - `mask` (ndarray H×W bool) — editable pixels
  - `kernel` (ndarray)
  - `points` (ndarray N×2)
  - `config` (dict) — target_gv, tol_pct, max_iters, patience, early_stop_*, use_gpu, block_size
- **Outputs**:
  - `run_optimization(image, mask, kernel, points, config) -> generator` — yields `IterResult` per iteration
  - `IterResult` fields: `iter`, `image`, `conv_map`, `monitor_gv`, `rmse`, `max_err`, `frac_within_tol`, `elapsed`
- **Key functions**: `run_optimization`
- **Notes**: prefer block/tile updates; avoid dense full-resolution matrix solves

---

### `logger.py`

- **Status**: not yet implemented
- **Purpose**: print per-iteration statistics and manage tqdm progress bars
- **Inputs**:
  - `result` (IterResult) — from `optimizer.run_optimization`
  - `backend` (str) — `"CPU"` or `"GPU"`
- **Outputs**:
  - `log_iter(result, backend)` — prints to stdout
  - `make_pbar(total, desc) -> tqdm` — returns configured progress bar
- **Key functions**: `log_iter`, `make_pbar`

---

### `outputs.py`

- **Status**: not yet implemented
- **Purpose**: save all per-iteration and final artefacts
- **Inputs**:
  - `result` (IterResult)
  - `original_image` (ndarray H×W) — for diff image
  - `output_dir` (str/Path)
  - `save_diff` (bool)
- **Outputs** (files written):
  - `output/edited_images/iter_{N:04d}.png`
  - `output/kernel_maps/iter_{N:04d}.png`
  - `output/monitor_csv/iter_{N:04d}.csv`
  - `output/plots/iter_{N:04d}.png`
  - `output/diff_images/iter_{N:04d}.png` (optional)
  - `output/iteration_summary.csv` (appended each iteration)
- **Key functions**: `save_iter_outputs`, `save_final_outputs`

---

### `main.py`

- **Status**: not yet implemented
- **Purpose**: CLI entry point; parse arguments; orchestrate all modules
- **Inputs** (CLI args):

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input_image` | str | required | Path to base PNG |
| `--sections_csv` | str | required | Path to sections.csv |
| `--points_csv` | str | required | Path to points.csv |
| `--output_dir` | str | `output` | Output directory |
| `--sigma` | float | required | Kernel sigma |
| `--kind` | str | `gaussian` | Kernel type |
| `--trunc` | float | `4.0` | Kernel truncation |
| `--target_gv` | float | `20.0` | Target gray value |
| `--tol_pct` | float | `10.0` | Tolerance % around target |
| `--max_iters` | int | `200` | Max iterations |
| `--save_every` | int | `1` | Save outputs every N iters |
| `--force_cpu` | flag | False | Disable GPU |
| `--block_size` | int | `16` | Tile/block size for optimizer |
| `--early_stop_rmse` | float | None | Stop if RMSE < threshold |
| `--early_stop_uniformity_std` | float | None | Stop if std < threshold |
| `--early_stop_all_within_tolerance` | flag | False | Stop when all points in tolerance |
| `--patience` | int | `10` | Iterations without improvement before stop |

- **Outputs**: calls all other modules; writes to `--output_dir`
- **Key functions**: `parse_args`, `main`

---

## Output Directory Structure

```
output/
├── edited_images/        # per-iteration edited input image
├── kernel_maps/          # per-iteration convolved output map
├── monitor_csv/          # per-iteration monitor point CSV
├── plots/                # per-iteration monitor point plot
├── diff_images/          # per-iteration diff vs original (optional)
├── iteration_summary.csv # one row per iteration, appended live
└── final/                # copies of last iteration outputs
```

---

## Dependencies

```
numpy
scipy          # CPU FFT convolution fallback
Pillow         # image I/O
pandas         # CSV I/O
matplotlib     # plots
tqdm           # progress bars
cupy           # optional GPU acceleration
```
