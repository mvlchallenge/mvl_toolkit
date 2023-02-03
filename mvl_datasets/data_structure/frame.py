import os

import numpy as np
from imageio import imread

from mvl_datasets.utils.geometry_utils import extend_array_to_homogeneous
from mvl_datasets.utils.image_utils import load_depth_map


class Frame:

    def __init__(self, dt):
        self.dt = dt
        self.pose = None
        self.idx = None
        self.rgb_file = None
        self.depth_map_file = None
        self.rgb_map = None
        self.depth_map = None
        self.pcl = None

    def get_rgb(self):
        if self.rgb_map is None:
            if not os.path.exists(self.rgb_file):
                raise FileNotFoundError(self.rgb_file)
            self.rgb_map = imread(self.rgb_file)
        return self.rgb_map

    def get_depth(self):
        if self.depth_map is None:
            if not os.path.exists(self.depth_map_file):
                raise FileNotFoundError(self.depth_map_file)
            self.depth_map = load_depth_map(self.depth_map_file)
        return self.depth_map

    def get_pcl(self):
        if self.pcl is None:
            pcl, color = self.dt.cam.project_pcl_from_depth_and_rgb_maps(
                color_map=self.get_rgb(),
                depth_map=self.get_depth(),
            )
            self.pcl = np.vstack((
                self.pose[:3, :] @ extend_array_to_homogeneous(pcl),
                color
            ))
        return self.pcl
