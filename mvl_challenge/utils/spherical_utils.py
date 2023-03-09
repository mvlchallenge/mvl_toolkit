import math

import numpy as np


class SphericalCamera:
    def __init__(self, shape):
        self.shape = shape
        self.compute_default_grids()

    def compute_default_grids(self):
        h, w = self.shape
        u = np.linspace(0, w - 1, w).astype(int)
        v = np.linspace(0, h - 1, h).astype(int)
        uu, vv = np.meshgrid(u, v)
        self.default_pixel = np.vstack((uu.flatten(), vv.flatten())).astype(np.int)
        self.default_bearings = uv2xyz(self.default_pixel, self.shape)

    def project_pcl_from_depth_and_rgb_maps(self, color_map, depth_map, scaler=1):
        from mvl_challenge.utils.image_utils import get_color_array
        color_pixels = get_color_array(color_map=color_map) / 255
        mask = depth_map.flatten() > 0
        pcl = self.default_bearings[:, mask] * scaler * get_color_array(color_map=depth_map)[0][mask]
        return pcl, color_pixels[:, mask]


def uv2xyz(uv, shape):
    """
    Projects uv vectors to xyz vectors (bearing vector)
    """
    sph = uv2sph(uv, shape)
    theta = sph[0]
    phi = sph[1]

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    return np.vstack((x, y, z))


def uv2sph(uv, shape):
    """
    Projects a set of uv points into spherical coordinates (theta, phi)
    """
    H, W = shape
    theta = 2 * np.pi * ((uv[0]) / W - 0.5)
    phi = np.pi * ((uv[1]) / H - 0.5)
    return np.vstack((theta, phi))


def sph2xyz(sph):
    """
    Projects spherical coordinates (theta, phi) to euclidean space xyz
    """
    theta = sph[:, 0]
    phi = sph[:, 1]

    x = math.cos(phi) * math.sin(theta)
    y = math.sin(phi)
    z = math.cos(phi) * math.cos(theta)

    return np.vstack((x, y, z))

#! Checked OK
def sph2uv(sph, shape):
    # H, W = shape
    # u = W * (sph[0]/(2*np.pi) + 0.5)
    # v = H * (sph[1]/np.pi + 0.5)
    # return np.floor(np.vstack((
    #     np.clip(u, 0, W-1),
    #     np.clip(v, 0, H-1)
    # ))).astype(int)
    theta_coord = sph[0]
    phi_coord = sph[1]
    u = np.clip(np.floor((0.5 * theta_coord / np.pi + 0.5) * shape[1] + 0.5), 0, shape[1] - 1)
    v = np.clip(np.floor((phi_coord / np.pi + 0.5) * shape[0]+0.5), 0, shape[0] - 1)
    return np.vstack([u, v]).astype(int)


def sphere_normalization(xyz):
    norm = np.linalg.norm(xyz, axis=0)
    return xyz/norm


#! Checked ok!
def phi_coords2xyz(phi_coords):
    """
    Returns 3D bearing vectors (on the unite sphere) from phi_coords
    """
    W = phi_coords.__len__()
    u = np.linspace(0, W - 1, W)
    theta_coords = (2 * np.pi * u / W) - np.pi
    bearings_y = np.sin(phi_coords)
    bearings_x = np.cos(phi_coords) * np.sin(theta_coords)
    bearings_z = np.cos(phi_coords) * np.cos(theta_coords)
    return np.vstack((bearings_x, bearings_y, bearings_z))


#! Checked ok!
def phi_coords2uv(phi_coord, shape=(512, 1024)):
    """
    Converts a set of phi_coordinates (2, W), defined by ceiling and floor boundaries encoded as 
    phi coordinates, into uv pixels 
    """
    H, W = shape
    u = np.linspace(0, W - 1, W)
    theta_coords = (2 * np.pi * u / W) - np.pi
    uv_c = sph2uv(np.vstack((theta_coords, phi_coord[0])), shape)
    uv_f = sph2uv(np.vstack((theta_coords, phi_coord[1])), shape)
    return uv_c, uv_f


#! Checked ok!
def uv2phi_coords(uv, shape=(512, 1024), type_bound='floor'):
    # _, idx, count = np.unique(uv[0], return_index=True, return_counts=True)
    u_coords = np.linspace(0, shape[1] - 1, shape[1]).astype(np.int16)
    v = []
    for u in u_coords:
        v_idx = np.where(uv[0] == u)[0]
        if type_bound == 'floor':
            v.append(np.max(uv[1, v_idx]))
        elif type_bound == 'ceiling':
            v.append(np.min(uv[1, v_idx]))
        else: 
            raise ValueError("wrong type_bound")
    
    phi_bon = (np.array(v) / shape[0] - 0.5) * np.pi
    return phi_bon

#! Checked ok!
def xyz2uv(xyz, shape=(512, 1024)):
    """
    Projects XYZ array into uv coord
    """
    xyz_n = xyz / np.linalg.norm(xyz, axis=0, keepdims=True)

    normXZ = np.linalg.norm(xyz[(0, 2), :], axis=0, keepdims=True)

    phi_coord = np.arcsin(xyz_n[1, :])
    theta_coord = np.sign(xyz[0, :]) * np.arccos(xyz[2, :] / normXZ)

    u = np.clip(np.floor((0.5 * theta_coord / np.pi + 0.5) * shape[1] + 0.5), 0, shape[1] - 1)
    v = np.clip(np.floor((phi_coord / np.pi + 0.5) * shape[0]+0.5), 0, shape[0] - 1)
    return np.vstack((u, v)).astype(int)