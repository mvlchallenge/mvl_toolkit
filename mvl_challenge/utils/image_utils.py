from imageio import imread
import numpy as np

def get_color_array(color_map):
    """
    returns an array (3, n) of the colors in image (H, W)
    """
    # ! This is the same solution by flatten every channel
    if len(color_map.shape) > 2:
        return np.vstack((color_map[:, :, 0].flatten(),
                          color_map[:, :, 1].flatten(),
                          color_map[:, :, 2].flatten()))
    else:
        return np.vstack((color_map.flatten(),
                          color_map.flatten(),
                          color_map.flatten()))


def load_depth_map(fpath):
    """Make sure the depth map has shape (H, W) but not (H, W, 1)."""
    depth_map = imread(fpath)
    if depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    return depth_map
