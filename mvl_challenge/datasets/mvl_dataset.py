import os
import json
from traceback import print_tb
from tqdm import tqdm
from mvl_challenge.utils.layout_utils import label_cor2ly_phi_coord
from mvl_challenge.utils.geometry_utils import eulerAnglesToRotationMatrix, extend_array_to_homogeneous
from mvl_challenge.utils.geometry_utils import tum_pose2matrix44
from mvl_challenge.utils.spherical_utils import phi_coords2xyz
from mvl_challenge.data_structure import Layout, CamPose
import numpy as np
import logging
from imageio import imread
from pyquaternion import Quaternion
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MVLDataset:
    def __init__(self, cfg):
        logging.info("Initializing MVL Dataset...")
        self.cfg = cfg
        self.set_paths()
        logging.info("MVL Dataset successfully initialized")

    def set_paths(self):
        # * Paths for MVLDataset
        self.DIR_GEOM_INFO = os.path.join(
            self.cfg.runners.mvl.data_dir, 'geometry_info')
        self.DIR_IMGS = os.path.join(
            self.cfg.runners.mvl.data_dir, 'img'
            )
        
        assert os.path.exists(self.DIR_GEOM_INFO), f"Not found geometry info {self.DIR_GEOM_INFO}"
        assert os.path.exists(self.DIR_IMGS), f"Not found img dir {self.DIR_IMGS}"
        
        # * Set main lists of files
        #! List of scene names
        assert os.path.exists(self.cfg.runners.mvl.scene_list), f"Not found scene list {self.cfg.runners.mvl.scene_list}"
        logging.info(f"Scene list: {self.cfg.runners.mvl.scene_list}")
        self.data_scenes = json.load(
            open(self.cfg.runners.mvl.scene_list))
        self.list_rooms = list(self.data_scenes.keys())
        self.list_frames = [fr for fr in self.data_scenes.values()]
        self.list_frames = [item for sublist in self.list_frames for item in sublist]


    def iter_list_ly(self):
        """Iterator for all room scene defined in this class
        """
        for room_scene in self.list_rooms:
            list_ly = self.get_list_ly(room_scene)
            yield list_ly
        
    def get_list_ly(self, room_scene):
        """
        Returns a list of Layout instances described by room_scene. 
        Scene name is the room name for the scene.
        By default it returns all scene names.  
        """
        
        scene_data = self.data_scenes[room_scene]

        self.list_ly = []
        for frame in tqdm(scene_data, desc=f"Loading mvl data scene {room_scene}..."):

            ly = Layout(self.cfg)
            ly.idx = os.path.splitext(frame)[0]

            ly.img_fn = os.path.join(self.DIR_IMGS, f"{ly.idx}.jpg")
            assert os.path.exists(ly.img_fn), f"Not found {ly.img_fn}" 
            # ! Loading geometry
            geom = json.load(
                open(os.path.join(self.DIR_GEOM_INFO, f"{ly.idx}.json")))

            self.set_geom_info(layout=ly, geom=geom)
            
            # ! Setting in WC
            ly.cam_ref = 'WC'

            self.list_ly.append(ly)            
        
        return self.list_ly

    def print_mvl_data_info(self):
        a = [dt.__len__() for dt in self.data_scenes.values()]
        logging.info(f"Total number of frames: {np.sum(a)}")
        logging.info(f"Total number of room scenes: {a.__len__()}")
        
    @staticmethod
    def set_geom_info(layout: Layout, geom):
        
        layout.pose = CamPose(layout.cfg)
        layout.pose.t = np.array(geom['translation'])  # * geom['scale']
        qx, qy, qz, qw = geom['quaternion']
        q = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)
        layout.pose.rot = q.rotation_matrix
        layout.pose.idx = layout.idx
        layout.camera_height = geom['cam_h']
        