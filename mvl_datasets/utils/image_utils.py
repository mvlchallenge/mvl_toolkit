from imageio import imread


def load_depth_map(fpath):
    """Make sure the depth map has shape (H, W) but not (H, W, 1)."""
    depth_map = imread(fpath)
    if depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    return depth_map
