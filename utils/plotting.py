import numpy as np
import matplotlib.pyplot as plt

def show_hdr(img, ax=None, mode="log", lo=1, hi=99.5, title=None,
             extent=None, horizon_row=None):
    """Display-only tone map. Input: raw or scaled linear radiance."""
    x = np.asarray(img, dtype=np.float64)
    pos = x[x > 0]
    floor = np.percentile(pos, 0.1) if pos.size else 1e-8

    if mode == "log":
        y = np.log10(np.maximum(x, floor))
    elif mode == "gamma":
        y = np.power(np.maximum(x, 0) / np.percentile(x, hi), 1 / 2.2)
    else:
        y = x

    vmin, vmax = np.percentile(y, [lo, hi])
    ax = ax or plt.gca()
    ax.imshow(y, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal", extent=extent)
    if horizon_row is not None:
        ax.axhline(horizon_row, color="tab:red", lw=0.5)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    return ax