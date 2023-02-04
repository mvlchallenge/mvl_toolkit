import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mvl_datasets.config.cfg import get_empty_cfg
from mvl_datasets.data_structure.frame import Frame
from mvl_datasets.utils.io_utils import read_trajectory
from mvl_datasets.utils.spherical_utils import SphericalCamera
from mvl_datasets.utils.vispy_utils import plot_color_plc


class RGBD_Dataset:
        
    @classmethod
    def from_args(clc, args):
        assert os.path.exists(args.scene_dir)
        
        cfg = get_empty_cfg()
        cfg.dataset = dict()
        cfg.dataset.scene_dir = args.scene_dir
        return clc.from_cfg(cfg)
    
    @classmethod
    def from_cfg(clc, cfg):   
        # MP3D-FPE dataset has a vo* directory
        vo_dir = glob.glob(os.path.join(cfg.dataset.scene_dir, 'vo_*'))
        if vo_dir.__len__() == 0: 
            # HM3D-MVL dataset
            dt = HM3D_MVL(cfg)
        else:
            # MP3D-FPE
            dt = MP3D_FPE(cfg)
        
        return dt  
    
    @classmethod
    def from_scene_dir(clc, scene_dir):
        assert os.path.exists(scene_dir)     
        cfg = get_empty_cfg()
        cfg.dataset = dict()
        cfg.dataset.scene_dir = scene_dir
        return clc.from_cfg(cfg)
        

                  
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_paths()
        self.load_data()
        self.cam = SphericalCamera(shape=cfg.get("resolution", (512, 1024)))
        logging.info(f"RGBD dataset loaded successfully")
        logging.info(f"Scene directory: {self.scene_dir}")
        logging.info(f"Scene name: {self.scene_name}")

    def load_data(self):
        # ! List of Kf
        self.set_list_of_frames()

        # ! List of camera poses
        self.load_camera_poses()

        # ! List of files
        self.rgb_files = [os.path.join(self.rgb_dir, f'{f}.{self.rgb_ext}') for f in self.kf_list]
        self.depth_files = [os.path.join(self.depth_dir, f'{f}.tiff') for f in self.kf_list]

    def load_camera_poses(self):
        """
        Load both GT  camera poses
        """

        # ! Loading GT camera poses
        gt_poses_file = os.path.join(
            self.scene_dir,
            'frm_ref.txt')

        assert os.path.isfile(
            gt_poses_file
        ), f'Cam pose file {gt_poses_file} does not exist'

        self.gt_poses = np.stack(
            list(read_trajectory(gt_poses_file).values()))[self.idx, :, :]

    def set_paths(self):
        self.scene_dir = Path(self.cfg.dataset.scene_dir).resolve().__str__()
        self.scene_name = "_".join(self.scene_dir.split("/")[-2:])
        # set rgb directory
        self.rgb_dir = os.path.join(self.scene_dir, "rgb")
        if not os.path.isdir(self.rgb_dir):
            raise FileExistsError(f"{self.rgb_dir}")

        # set depth directory
        self.depth_dir = os.path.join(self.scene_dir, "depth", "tiff")
        if not os.path.isdir(self.depth_dir):
            raise FileExistsError(f"{self.depth_dir}")

    def set_list_of_frames(self):

        raise NotImplementedError(
            """
        Setting the list of frames in the data depends on the dataset. 
        MP3D-FPE uses a set of estimated key-frames, which are defined in a vo-* directory. 
        While HM3D-MVL uses directly the available images in rgb/ directory.
        """
        )

    def get_list_frames(self):
        list_fr = []
        for rgb_fn, depth_fn, pose in tqdm(zip(self.rgb_files, self.depth_files, self.gt_poses), desc="Loading frames..."):
            fr = Frame(self)
            fr.rgb_file = rgb_fn
            fr.depth_map_file = depth_fn
            fr.idx = int(Path(rgb_fn).stem)
            fr.pose = pose
            list_fr.append(fr)
        return list_fr


class MP3D_FPE(RGBD_Dataset):
    def __init__(self, cfg):
        self.rgb_ext = 'png'
        super().__init__(cfg)

    def set_list_of_frames(self):
        self.vo_dir = glob.glob(os.path.join(self.scene_dir, 'vo_*'))[0]
        list_keyframe_fn = os.path.join(self.vo_dir, 'keyframe_list.txt')
        assert os.path.exists(list_keyframe_fn), f"{list_keyframe_fn}"

        with open(list_keyframe_fn, 'r') as f:
            self.kf_list = sorted([int(kf) for kf in f.read().splitlines()])
        self.idx = np.array(self.kf_list) - 1


class HM3D_MVL(RGBD_Dataset):
    def __init__(self, cfg):
        self.rgb_ext = "jpg"
        super().__init__(cfg)

    def set_list_of_frames(self):
        self.kf_list = sorted([int(os.path.basename(f).split(".")[0]) for f in os.listdir(self.rgb_dir)])
        self.idx = np.array(self.kf_list)

    
def get_default_args():
    parser = argparse.ArgumentParser()

    # * Input Directory (-s)
    parser.add_argument(
        '-scene_dir',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES/2t7WUuJeko7/0",
        type=str,
        help='Input Directory (scene directory defined until version scene)'
    )

    args = parser.parse_args()

    return args


def main(args):
    dt = MP3D_FPE.from_args(args)
    list_fr = dt.get_list_frames()
    pcl  = np.hstack([fr.get_pcl() for fr in list_fr[:4]])
    plot_color_plc(points=pcl[0:3, :].T, color=pcl[3:].T)


if __name__ == '__main__':
    args = get_default_args()
    main(args)
