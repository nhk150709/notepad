#!/usr/bin/env python3
"""
electroplating_sim.py
=====================
Fast engineering approximation for estimating electroplating thickness
uniformity over a panel with a perforated insulating shield.

USAGE
-----
    pip install numpy scipy matplotlib Pillow
    python electroplating_sim.py --config config.json
    python electroplating_sim.py --config config.json --output my_results/

MODELING APPROACH
-----------------
Effective field at the panel is approximated as:

    F(x, y)  =  (ApertureMask  *  DispersionKernel)(x, y)

where * denotes 2-D convolution. This represents current lines
spreading laterally as they travel from the shield apertures to the
panel surface.  The kernel (Gaussian or Lorentzian) is shift-invariant,
so the convolution is computed ONCE on a working grid fixed to the
shield/world frame.  Time integration then just samples the pre-computed
field F at positions shifted by the panel displacement δ(t):

    E(x_p, y_p, t)  =  F(x_p + δ(t),  y_p)
    δ(t)            =  A · sin(2π t / T)

Thickness map:
    T_avg(x_p, y_p)  =  (1/N) Σ_t  E(x_p, y_p, t)
    T_rel (%)        =  T_avg / mean(T_avg) × 100

ASSUMPTIONS & LIMITATIONS
--------------------------
* Approximate field model — NOT a full FEM electrostatic solver.
* Gaussian / Lorentzian kernel is empirical; sigma is a tunable parameter.
* Uniform anode (constant current density); no edge or robber effects.
* One-sided model; constant cathode efficiency; no mass-transport limits.
* Absolute thickness (µm) requires calibration data not yet available.
* Shield is stationary; only the panel oscillates in X.
"""

import json
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


# =============================================================================
# CONFIGURATION
# =============================================================================

_DEFAULTS = {
    'metal': 'Cu',
    'anode_to_panel_gap': 30.0,
    'shield_thickness': 5.0,
    'shield_offset_x': 0.0,
    'shield_offset_y': 0.0,
    'hex_orientation': 'vertical_stagger',
    'blocked_holes': [],
    'image_mask_file': None,
    'kernel_type': 'gaussian',
    'kernel_sigma': None,           # auto: 0.5 × shield_to_panel_gap
    'polarization_exponent': None,  # None = disabled
    'oscillation_type': 'sinusoidal',
    'n_snapshot_steps': 4,
    'enable_animation': False,
    'animation_fps': 10,
    'output_folder': 'output',
}

_REQUIRED = [
    'panel_width', 'panel_height', 'grid_step',
    'current_density', 'plating_time_s', 'cathode_efficiency',
    'oscillation_amplitude', 'oscillation_period',
    'shield_to_panel_gap', 'shield_width', 'shield_height',
    'hole_diameter', 'hole_pitch', 'n_time_steps',
]


def load_config(path):
    """Load JSON config, fill defaults, validate required keys."""
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    # Remove comment key if present
    cfg.pop('_comment', None)
    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, v)
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Required config key(s) missing: {missing}")
    # Auto kernel sigma
    if cfg['kernel_sigma'] is None:
        cfg['kernel_sigma'] = cfg['shield_to_panel_gap'] * 0.5
        print(f"[INFO] kernel_sigma auto-set to {cfg['kernel_sigma']:.1f} mm "
              f"(= 0.5 × shield_to_panel_gap)")
    return cfg


# =============================================================================
# GRID
# =============================================================================

def build_grid(cfg):
    """
    Build panel grid and extended working grid.

    World / shield frame is fixed.  Panel at rest occupies
    x ∈ [0, W], y ∈ [0, H].  When displaced by δ it occupies
    x ∈ [δ, W+δ].

    The working grid shares the same cell structure as the panel
    grid but is extended by margin cells in every direction so
    that the convolution and all displacement samples stay in bounds.
    """
    step = cfg['grid_step']
    W, H = cfg['panel_width'], cfg['panel_height']
    amp  = cfg['oscillation_amplitude']
    sig  = cfg['kernel_sigma']

    # Panel cell centres (half-cell offset from edges)
    px = np.arange(step / 2, W, step)
    py = np.arange(step / 2, H, step)
    panel_X, panel_Y = np.meshgrid(px, py, indexing='ij')
    nx_p, ny_p = panel_X.shape

    # Margin: enough for max displacement + kernel tail (4σ captures >99.99%)
    mx = int(np.ceil((amp + 4 * sig) / step)) + 2
    my = int(np.ceil(4 * sig / step)) + 2

    # Extend panel grid outward
    left_x  = px[0]  - np.arange(mx, 0, -1) * step
    right_x = px[-1] + np.arange(1, mx + 1)  * step
    bot_y   = py[0]  - np.arange(my, 0, -1) * step
    top_y   = py[-1] + np.arange(1, my + 1)  * step

    wx = np.concatenate([left_x,  px, right_x])
    wy = np.concatenate([bot_y,   py, top_y])
    work_X, work_Y = np.meshgrid(wx, wy, indexing='ij')

    return dict(
        step=step, W=W, H=H,
        panel_X=panel_X, panel_Y=panel_Y,
        panel_x=px, panel_y=py,
        work_X=work_X, work_Y=work_Y,
        work_x=wx, work_y=wy,
        i0=mx, j0=my,       # index of panel origin in working grid
        nx_p=nx_p, ny_p=ny_p,
    )


# =============================================================================
# APERTURE MASK
# =============================================================================

def build_procedural_mask(cfg, grid):
    """
    Generate binary aperture mask on the working grid.
    1 = open (current passes through), 0 = blocked (insulating PVC).

    Hexagonal hole array with two orientation modes:
      vertical_stagger   — columns spaced pitch·√3/2 in x,
                           rows spaced pitch in y,
                           odd columns offset pitch/2 in y.
                           Nearest-neighbour distance = pitch.
      horizontal_stagger — rows spaced pitch·√3/2 in y,
                           columns spaced pitch in x,
                           odd rows offset pitch/2 in x.
    """
    work_X, work_Y = grid['work_X'], grid['work_Y']
    work_x, work_y = grid['work_x'], grid['work_y']
    step  = grid['step']
    W, H  = grid['W'], grid['H']

    pitch  = cfg['hole_pitch']
    radius = cfg['hole_diameter'] / 2.0
    orient = cfg['hex_orientation']

    # Shield centre in world frame
    scx = W / 2 + cfg['shield_offset_x']
    scy = H / 2 + cfg['shield_offset_y']
    sw, sh = cfg['shield_width'], cfg['shield_height']
    xs_lo, xs_hi = scx - sw / 2, scx + sw / 2
    ys_lo, ys_hi = scy - sh / 2, scy + sh / 2

    if orient == 'vertical_stagger':
        cdx = pitch * np.sqrt(3) / 2   # column spacing in x
        rdy = pitch                     # row spacing in y
        stagger_y = rdy / 2             # offset for odd columns
    else:  # horizontal_stagger
        cdx = pitch
        rdy = pitch * np.sqrt(3) / 2
        stagger_x = cdx / 2

    # Blocked holes look-up set (world-frame coordinates)
    blocked = {(round(float(b[0]), 2), round(float(b[1]), 2))
               for b in cfg.get('blocked_holes', [])}

    # --- Generator of hole centres ---
    def _centers_vertical():
        ci = int(np.floor((xs_lo - scx) / cdx)) - 1
        while True:
            hx = scx + ci * cdx
            if hx > xs_hi + cdx:
                break
            y_off = (ci % 2) * stagger_y
            ri = int(np.floor((ys_lo - scy - y_off) / rdy)) - 1
            while True:
                hy = scy + ri * rdy + y_off
                if hy > ys_hi + rdy:
                    break
                yield hx, hy
                ri += 1
            ci += 1

    def _centers_horizontal():
        ri = int(np.floor((ys_lo - scy) / rdy)) - 1
        while True:
            hy = scy + ri * rdy
            if hy > ys_hi + rdy:
                break
            x_off = (ri % 2) * stagger_x
            ci = int(np.floor((xs_lo - scx - x_off) / cdx)) - 1
            while True:
                hx = scx + ci * cdx + x_off
                if hx > xs_hi + cdx:
                    break
                yield hx, hy
                ci += 1
            ri += 1

    centers = _centers_vertical if orient == 'vertical_stagger' \
              else _centers_horizontal

    # --- Rasterise holes onto working grid ---
    mask = np.zeros(work_X.shape, dtype=bool)
    r2 = radius ** 2
    n_holes = 0

    for hx, hy in centers():
        key = (round(hx, 2), round(hy, 2))
        if key in blocked:
            continue
        n_holes += 1
        # Bounding-box indices in working grid
        ix0 = max(0, int(np.searchsorted(work_x, hx - radius - step)))
        ix1 = min(len(work_x), int(np.searchsorted(work_x, hx + radius + step)) + 1)
        iy0 = max(0, int(np.searchsorted(work_y, hy - radius - step)))
        iy1 = min(len(work_y), int(np.searchsorted(work_y, hy + radius + step)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        sX = work_X[ix0:ix1, iy0:iy1]
        sY = work_Y[ix0:ix1, iy0:iy1]
        mask[ix0:ix1, iy0:iy1] |= ((sX - hx) ** 2 + (sY - hy) ** 2 <= r2)

    print(f"[INFO] Procedural mask: {n_holes} holes on "
          f"{mask.shape[0]}×{mask.shape[1]} working grid")
    return mask.astype(np.float32)


def load_image_mask(cfg, grid):
    """
    Load a PNG aperture mask and resample it to the working grid.
    Convention: white (>128) = open, black (≤128) = blocked.
    The PNG is assumed to cover the entire working grid extent.
    """
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERROR] Pillow required for image mask.  "
                 "Install with:  pip install Pillow")

    img = Image.open(cfg['image_mask_file']).convert('L')
    nx_w, ny_w = grid['work_X'].shape   # (nx, ny) with indexing='ij'
    # PIL resize((width, height)) — width = nx (x direction), height = ny
    img_r = img.resize((nx_w, ny_w), Image.LANCZOS)
    arr = np.array(img_r, dtype=np.float32)   # PIL → shape (ny, nx) = (height, width)
    arr = arr.T                                # → (nx, ny)
    print(f"[INFO] Image mask loaded: {cfg['image_mask_file']}  "
          f"→ {arr.shape} working grid")
    return (arr > 128).astype(np.float32)


# =============================================================================
# DISPERSION KERNEL
# =============================================================================

def build_kernel(cfg, grid):
    """
    Build 2-D dispersion kernel representing lateral current spreading
    from shield apertures to the panel surface.

    Gaussian:   K(r) = exp(−r²/(2σ²))
    Lorentzian: K(r) = 1 / (1 + r²/σ²)

    Both are normalised to unit sum over the discrete grid.
    σ (kernel_sigma) is the primary tunable parameter.
    Physically it scales with the shield-to-panel gap; larger σ gives
    more lateral spreading and therefore better uniformity but lower
    current contrast between open and blocked regions.
    """
    step  = grid['step']
    sigma = cfg['kernel_sigma']
    ktype = cfg.get('kernel_type', 'gaussian')

    half = int(np.ceil(4 * sigma / step)) * step   # kernel half-width
    kv = np.arange(-half, half + step, step)
    KX, KY = np.meshgrid(kv, kv, indexing='ij')
    r2 = KX ** 2 + KY ** 2

    if ktype == 'gaussian':
        K = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
    elif ktype == 'lorentzian':
        K = (1.0 / (1.0 + r2 / sigma ** 2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown kernel_type '{ktype}'.  "
                         "Use 'gaussian' or 'lorentzian'.")
    K /= K.sum()
    return K


# =============================================================================
# OSCILLATION
# =============================================================================

def displacement(t, cfg):
    """
    Panel x-displacement at time t (mm).
    Panel moves; shield and anode are stationary.
    """
    A = cfg['oscillation_amplitude']
    T = cfg['oscillation_period']
    otype = cfg.get('oscillation_type', 'sinusoidal')

    if otype == 'sinusoidal':
        return A * np.sin(2 * np.pi * t / T)
    elif otype == 'triangular':
        p = (t % T) / T
        if   p < 0.25: return  A * 4 * p
        elif p < 0.75: return  A * (2 - 4 * p)
        else:          return  A * (4 * p - 4)
    else:
        raise ValueError(f"Unknown oscillation_type '{otype}'.  "
                         "Use 'sinusoidal' or 'triangular'.")


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(cfg, grid, mask, kernel):
    """
    Compute time-averaged relative thickness distribution.

    Key optimisation: because the dispersion kernel K is shift-invariant,
    the convolution F = mask ⊛ K is computed ONCE on the full working grid.
    Each time step then only requires extracting a sub-region of F at the
    shifted panel position — much faster than convolving at every step.

    Returns
    -------
    thickness_map : ndarray (nx_p, ny_p) — relative thickness in %
    snapshots     : list of (time_s, array) tuples
    """
    step   = grid['step']
    i0, j0 = grid['i0'], grid['j0']
    nx_p, ny_p = grid['nx_p'], grid['ny_p']
    n_steps  = cfg['n_time_steps']
    T_osc    = cfg['oscillation_period']
    n_snaps  = cfg['n_snapshot_steps']
    pol      = cfg.get('polarization_exponent')   # None → disabled

    # --- Pre-compute convolved field (once) ---
    print(f"[INFO] Convolving mask {mask.shape} ⊛ kernel {kernel.shape} …")
    F = fftconvolve(mask, kernel, mode='same')
    F = np.maximum(F, 0.0)   # no negative current

    # --- Time integration over one full oscillation period ---
    # (valid when total plating time >> oscillation period, which holds:
    #  7200 s >> 30 s, so ~240 complete cycles → steady-state average)
    times     = np.linspace(0, T_osc, n_steps, endpoint=False)
    snap_idx  = set(np.linspace(0, n_steps - 1, n_snaps, dtype=int).tolist())
    accum     = np.zeros((nx_p, ny_p), dtype=np.float64)
    snapshots = []

    for k, t in enumerate(times):
        d     = displacement(t, cfg)
        d_idx = int(round(d / step))   # nearest-cell shift

        il = i0 + d_idx
        ir = il + nx_p
        jl = j0
        jr = jl + ny_p

        # Safety: warn if margin is too small (should not happen with default config)
        if il < 0 or ir > F.shape[0]:
            print(f"[WARN] Step {k}: panel shift ({d:.1f} mm) exceeds working-grid "
                  f"margin; clamping.  Consider increasing kernel_sigma.")
            il = max(0, il)
            ir = min(F.shape[0], ir)

        pf = F[il:ir, jl:jr]

        # Handle shape mismatch from clamping
        if pf.shape != (nx_p, ny_p):
            tmp = np.zeros((nx_p, ny_p), dtype=np.float64)
            tmp[:pf.shape[0], :pf.shape[1]] = pf
            pf = tmp

        # Optional polarisation / Butler-Volmer smoothing
        if pol is not None:
            pf = np.power(np.maximum(pf, 1e-12), pol)

        accum += pf
        if k in snap_idx:
            snapshots.append((t, pf.copy()))

    # --- Normalise to mean = 100 % ---
    tmap   = accum / n_steps
    mean_v = tmap.mean()
    if mean_v > 0:
        factor    = 100.0 / mean_v
        tmap     *= factor
        snapshots = [(t, s * factor) for t, s in snapshots]
    else:
        print("[WARN] Zero mean field — check mask and kernel settings.")

    return tmap, snapshots


# =============================================================================
# STATISTICS
# =============================================================================

def compute_stats(arr):
    """Return basic uniformity statistics for a thickness map."""
    flat = arr.ravel()
    m = float(flat.mean())
    s = float(flat.std())
    return dict(
        mean=m,
        std=s,
        min=float(flat.min()),
        max=float(flat.max()),
        range=float(flat.max() - flat.min()),
        cv_pct=float(s / m * 100) if m > 0 else 0.0,
    )


# =============================================================================
# VISUALISATION
# =============================================================================

def _clim(st):
    """Symmetric colour limits centred at 100 %, 3σ range, clipped to [50, 150]."""
    half = max(st['std'] * 3.0, 5.0)
    return max(50.0, 100.0 - half), min(150.0, 100.0 + half)


def _save_heatmap(arr, px, py, title, path, vmin, vmax, cmap='RdYlGn'):
    """
    Save thickness heatmap.
    arr has shape (nx, ny) with indexing='ij' (x=rows, y=cols).
    pcolormesh expects C[row=y, col=x], so we pass arr.T.
    Axes: horizontal=X (panel width direction), vertical=Y (height direction).
    """
    fig, ax = plt.subplots(figsize=(7, 7.5))
    im = ax.pcolormesh(px, py, arr.T,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Relative thickness (%)', fontsize=10)
    ax.set_xlabel('X position (mm)', fontsize=10)
    ax.set_ylabel('Y position (mm)', fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_mask_preview(mask, grid, cfg, path):
    """Save working-grid aperture mask with panel-outline overlay."""
    wx, wy = grid['work_x'], grid['work_y']
    W, H   = grid['W'], grid['H']
    amp    = cfg['oscillation_amplitude']

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(wx, wy, mask.T, cmap='gray', vmin=0, vmax=1, shading='auto')

    # Panel outline at rest (red)
    rect0 = plt.Rectangle((0, 0), W, H, fill=False,
                           edgecolor='red', lw=2.0, label='Panel at rest')
    ax.add_patch(rect0)

    # Panel outline at ±amplitude (orange dashed)
    for dx in [amp, -amp]:
        r = plt.Rectangle((dx, 0), W, H, fill=False,
                           edgecolor='orange', lw=1.2, linestyle='--')
        ax.add_patch(r)
    # Dummy handle for legend
    ax.plot([], [], color='orange', lw=1.2, linestyle='--',
            label=f'Panel at ±{amp:.0f} mm')

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('Aperture mask (white = open, black = blocked)\n'
                 'Red = panel at rest;  Orange = ±amplitude')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _save_animation(cfg, grid, snapshots, out_dir, ts, vmin, vmax):
    """Save GIF animation (only called when enable_animation = true)."""
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("[WARN] FuncAnimation not available; skipping animation.")
        return

    px, py = grid['panel_x'], grid['panel_y']
    fps    = cfg.get('animation_fps', 10)

    fig, ax = plt.subplots(figsize=(7, 7.5))
    mesh = ax.pcolormesh(px, py, snapshots[0][1].T,
                         cmap='RdYlGn', vmin=vmin, vmax=vmax, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Relative thickness (%)')
    ax.set_aspect('equal')
    ttl = ax.set_title('')

    def _update(i):
        t, s = snapshots[i]
        mesh.set_array(s.T.ravel())
        ttl.set_text(f't = {t:.1f} s   δ = {displacement(t, cfg):+.1f} mm')
        return mesh, ttl

    anim = FuncAnimation(fig, _update, frames=len(snapshots),
                         interval=1000 / fps, blit=False)
    p = os.path.join(out_dir, f'animation_{ts}.gif')
    try:
        anim.save(p, writer='pillow', fps=fps)
        print(f"[OUT] Animation      → {p}")
    except Exception as e:
        print(f"[WARN] Animation save failed: {e}")
    plt.close(fig)


def save_all(cfg, grid, mask, tmap, snapshots, st, out_dir):
    """Write all output files."""
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    px, py = grid['panel_x'], grid['panel_y']
    vmin, vmax = _clim(st)

    # 1 — Time-averaged thickness heatmap
    p = os.path.join(out_dir, f'thickness_avg_{ts}.png')
    _save_heatmap(tmap, px, py,
                  (f"Time-averaged relative thickness (%)\n"
                   f"std={st['std']:.2f}%  "
                   f"min={st['min']:.1f}%  "
                   f"max={st['max']:.1f}%  "
                   f"CV={st['cv_pct']:.2f}%"),
                  p, vmin, vmax)
    print(f"[OUT] Avg map        → {p}")

    # 2 — Instantaneous snapshots
    for i, (t, snap) in enumerate(snapshots):
        ss = compute_stats(snap)
        d  = displacement(t, cfg)
        p  = os.path.join(out_dir, f'snapshot_{i:02d}_{ts}.png')
        _save_heatmap(snap, px, py,
                      (f"Instantaneous field  t={t:.1f} s  δ={d:+.1f} mm\n"
                       f"std={ss['std']:.2f}%"),
                      p, vmin, vmax)
        print(f"[OUT] Snapshot {i:02d}     → {p}  (t={t:.1f}s, δ={d:+.1f}mm)")

    # 3 — Aperture mask preview
    p = os.path.join(out_dir, f'mask_preview_{ts}.png')
    _save_mask_preview(mask, grid, cfg, p)
    print(f"[OUT] Mask preview   → {p}")

    # 4 — Statistics JSON
    p = os.path.join(out_dir, f'statistics_{ts}.json')
    with open(p, 'w') as f:
        json.dump({'timestamp': ts, 'config': cfg, 'statistics': st},
                  f, indent=2)
    print(f"[OUT] Statistics     → {p}")

    # 5 — Console summary
    print()
    print("=" * 58)
    print("  ELECTROPLATING SHIELD SIMULATION — RESULTS")
    print("=" * 58)
    print(f"  Panel:            {cfg['panel_width']} × {cfg['panel_height']} mm")
    print(f"  Grid step:        {cfg['grid_step']} mm")
    print(f"  Hole Ø / pitch:   {cfg['hole_diameter']} / {cfg['hole_pitch']} mm")
    print(f"  Shield gap:       {cfg['shield_to_panel_gap']} mm")
    print(f"  Kernel:           {cfg['kernel_type']}  σ = {cfg['kernel_sigma']:.1f} mm")
    print(f"  Oscillation:      ±{cfg['oscillation_amplitude']} mm  "
          f"T = {cfg['oscillation_period']} s  ({cfg['oscillation_type']})")
    print(f"  Time steps:       {cfg['n_time_steps']}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Mean thickness:   {st['mean']:.2f}%  (normalised to 100 %)")
    print(f"  Std deviation:    {st['std']:.2f}%")
    print(f"  Min / Max:        {st['min']:.1f}% / {st['max']:.1f}%")
    print(f"  Range:            {st['range']:.1f}%")
    print(f"  CV (std/mean):    {st['cv_pct']:.2f}%")
    print("=" * 58)

    # 6 — Optional animation
    if cfg.get('enable_animation'):
        _save_animation(cfg, grid, snapshots, out_dir, ts, vmin, vmax)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Electroplating shield uniformity simulator  (fast, kernel-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python electroplating_sim.py\n"
            "  python electroplating_sim.py --config config.json --output results/\n"
        ),
    )
    ap.add_argument('--config', default='config.json',
                    help='Path to JSON config file (default: config.json)')
    ap.add_argument('--output', default=None,
                    help='Output folder override (overrides config output_folder)')
    args = ap.parse_args()

    print("[INFO] Electroplating shield simulation starting …")
    cfg = load_config(args.config)
    if args.output:
        cfg['output_folder'] = args.output

    grid = build_grid(cfg)
    nx_p, ny_p = grid['nx_p'], grid['ny_p']
    wgx, wgy   = grid['work_X'].shape
    print(f"[INFO] Panel grid:   {nx_p} × {ny_p} cells  "
          f"({grid['W']} × {grid['H']} mm,  step = {grid['step']} mm)")
    print(f"[INFO] Working grid: {wgx} × {wgy} cells")

    if cfg.get('image_mask_file'):
        mask = load_image_mask(cfg, grid)
    else:
        mask = build_procedural_mask(cfg, grid)
    print(f"[INFO] Open-area fraction: {mask.mean() * 100:.1f}%")

    kernel = build_kernel(cfg, grid)
    print(f"[INFO] Kernel: {cfg['kernel_type']},  "
          f"σ = {cfg['kernel_sigma']:.1f} mm,  "
          f"size = {kernel.shape[0]} × {kernel.shape[1]}")

    tmap, snapshots = run_simulation(cfg, grid, mask, kernel)
    st = compute_stats(tmap)

    save_all(cfg, grid, mask, tmap, snapshots, st, cfg['output_folder'])
    print("[INFO] Done.")


if __name__ == '__main__':
    main()
