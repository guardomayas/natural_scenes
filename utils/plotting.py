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

def plot_examples(imgs, n_rows, n_cols, random_seed=0, **kwargs):
    n_samples = n_rows * n_cols
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(imgs), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for ax, i in zip(axes.flat[:n_samples], idx):
        show_hdr(imgs[i, :, :], ax=ax, **kwargs)

    for ax in axes.flat[n_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes