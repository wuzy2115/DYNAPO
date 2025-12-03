import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_image_grid(
    images, rows, cols, directions=None, imsize=(2, 2), title=None, show=True
):
    fig, axs = plt.subplots(
        rows,
        cols,
        gridspec_kw={"wspace": 0, "hspace": 0},
        squeeze=True,
        figsize=(rows * imsize[0], cols * imsize[1]),
    )
    for i, image in enumerate(images):
        axs[i % rows][i // rows].axis("off")
        if directions is not None:
            axs[i % rows][i // rows].arrow(
                32,
                32,
                directions[i][0] * 16,
                directions[i][1] * 16,
                color="red",
                length_includes_head=True,
                head_width=2.0,
                head_length=1.0,
            )
        axs[i % rows][i // rows].imshow(image, aspect="auto")
    plt.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        fig.suptitle(title, fontsize=12)
    if show:
        plt.show()
    return fig


def show_save(save_path, show=True, save=False):
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


# Custom colormap from LaTeX attention visualization (reversed for uncertainty: low=teal, high=red)
_custom_colors_hex = [
    '5fbaa8',  # teal (low uncertainty)
    '88cfa4',  # green-blue
    'b2e0a2',  # light green
    'd7ef9b',  # yellow-green
    'eff8a6',  # light yellow-green
    'ffffbf',  # cream/yellow
    'feeb9e',  # light yellow
    'fdd380',  # yellow-orange
    'fdb466',  # light orange
    'f88d51',  # orange
    'f06744',  # red/orange (high uncertainty)
]
_custom_colors_rgb = [tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)) for h in _custom_colors_hex]
custom_cmap = LinearSegmentedColormap.from_list('custom_uncertainty', _custom_colors_rgb)


def color_tensor(tensor: torch.Tensor, cmap, norm=False):
    if norm:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if cmap == 'custom_uncertainty':
        map = custom_cmap
    else:
        map = plt.cm.get_cmap(cmap)
    tensor = torch.tensor(map(tensor.cpu().numpy()))[..., :3]
    return tensor
