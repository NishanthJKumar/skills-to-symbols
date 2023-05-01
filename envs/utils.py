import matplotlib
matplotlib.use("Agg")

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import os

def get_asset_path(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    return os.path.join(asset_dir_path, asset_name)

def fig2data(fig, dpi=150):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4, ))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data

def draw_token(token_image, x, y, ax, zoom=1.):
    oi = OffsetImage(token_image, zoom=zoom)
    box = AnnotationBbox(oi, (x, y), frameon=False)
    ax.add_artist(box)
    return box
