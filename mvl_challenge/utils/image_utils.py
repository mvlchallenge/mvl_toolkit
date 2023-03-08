from imageio import imread
import numpy as np
from mvl_challenge.utils.spherical_utils import phi_coords2xyz, xyz2uv

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


def draw_boundaries_uv(image, boundary_uv, color=(0, 1, 0), size=2):
    if image.shape.__len__() == 3:
        for i in range(size):
            image[(boundary_uv[1]+i) % image.shape[0], boundary_uv[0], :] = np.array(color)
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0], :] = np.array(color)
    else:
        for i in range(size):
            image[(boundary_uv[1]+i) % image.shape[0], boundary_uv[0]] = 255
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0]] = 255

    return image


def draw_boundaries_phi_coord(image, phi_coord, color=(0, 255, 0), size=2):

    # ! Compute bearings
    bearings_ceiling = phi_coords2xyz(
        phi_coords=phi_coord[0, :])
    bearings_floor = phi_coords2xyz(
        phi_coords=phi_coord[1, :])

    uv_ceiling = xyz2uv(bearings_ceiling)
    uv_floor = xyz2uv(bearings_floor)

    draw_boundaries_uv(
        image=image,
        boundary_uv=uv_ceiling,
        color=color,
        size=size
    )
    draw_boundaries_uv(
        image=image,
        boundary_uv=uv_floor,
        color=color,
        size=size
    )

    return image
