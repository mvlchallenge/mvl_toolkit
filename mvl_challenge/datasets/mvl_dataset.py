import os
import json
from traceback import print_tb
from tqdm import tqdm
from mvl_challenge.utils.layout_utils import label_cor2ly_phi_coord
from mvl_challenge.utils.geometry_utils import eulerAnglesToRotationMatrix, extend_array_to_homogeneous
from mvl_challenge.utils.geometry_utils import tum_pose2matrix44
from mvl_challenge.utils.spherical_utils import phi_coords2xyz
from mvl_challenge.data_structure import Layout, CamPose
from mvl_challenge.models.layout_model import LayoutModel
from mvl_challenge.utils.layout_utils import filter_out_noisy_layouts
import numpy as np
import logging
from imageio import imread
from pyquaternion import Quaternion
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch


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
        self.cfg._room_scene = room_scene
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
        

class MVImageLayout(data.Dataset):
    def __init__(self, list_data):
        self.data = list_data #[(img_fn, idx),...]

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        image_fn = self.data[idx][0]
        assert os.path.exists(image_fn)
        img = np.array(Image.open(image_fn), np.float32)[..., :3] / 255.
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        return dict(images=x, idx=self.data[idx][1])


def estimate_within_list_ly(list_ly, model: LayoutModel):
    """
    Estimates phi_coord (layout boundaries) for all ly defined in list_ly using the passed model instance
    """
    cfg = model.cfg
    layout_dataloader = DataLoader(
        MVImageLayout([(ly.img_fn, ly.idx) for ly in list_ly]),
        batch_size=cfg.runners.mvl.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.runners.mvl.num_workers,
        pin_memory=True if model.device != 'cpu' else False,
        worker_init_fn=lambda x: np.random.seed(),
    )
    model.net.eval()
    evaluated_data = {}
    for x in tqdm(layout_dataloader, desc=f"Estimating layout..."):
        with torch.no_grad():
            y_bon_, y_cor_ = model.net(x['images'].to(model.device))
            # y_bon_, y_cor_ = net(x[0].to(device))
        for y_, cor_, idx in zip(y_bon_.cpu(), y_cor_.cpu(), x['idx']):
            data = np.vstack((y_, cor_))
            evaluated_data[idx] = data

    [ly.recompute_data(phi_coord=evaluated_data[ly.idx]) for ly in list_ly]

        
def iter_mvl_room_scenes(model:LayoutModel, dataset: MVLDataset):
    """
    Creates a generator which yields a list of layout from a defined 
    MVL dataset and estimates layout in it.
    """
    
    dataset.print_mvl_data_info()
    cfg = dataset.cfg
    for room_scene in tqdm(dataset.list_rooms, desc="Reading MVL rooms..."):
        list_ly = dataset.get_list_ly(room_scene=room_scene)

        # ! Overwrite phi_coord within the list_ly by the estimating new layouts.
        estimate_within_list_ly(list_ly, model)
        filter_out_noisy_layouts(
            list_ly=list_ly,
            max_room_factor_size=cfg.runners.mvl.max_room_factor_size
        )        
        yield list_ly
