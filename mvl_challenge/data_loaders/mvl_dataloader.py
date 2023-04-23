import os
import numpy as np
import pathlib
import torch.utils.data as data
from PIL import Image
import json
import torch
import logging

  
class ListLayout:
    def __init__(self, list_ly):
        self.data = list_ly
    
    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        # ! iteration per each self.data 
        img = self.data[idx].img.transpose([2, 0, 1]).copy()
        x = torch.FloatTensor(np.array(img / img.max()))
        return dict(images=x, idx=self.data[idx].idx)


class MVLDataLoader(data.Dataset):
    '''
    Dataloader that handles MLC dataset format.
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        
        # ! List of scenes defined in a list file
        if cfg.get('scene_list', '') == '':
            # ! Reading from available labels data       
            self.raw_data = os.listdir(
                os.path.join(self.cfg.data_dir.labels_dir, self.cfg.label)
                )
            self.list_frames = self.raw_data
            self.list_rooms = None
        else:
            assert os.path.exists(self.cfg.scene_list)
            self.raw_data = json.load(open(self.cfg.scene_list))
            self.list_rooms = list(self.raw_data.keys())
            self.list_frames = [self.raw_data[room] for room in self.list_rooms]
            self.list_frames = [item for sublist in self.list_frames for item in sublist]
    
        if cfg.get('size', -1) < 0:
            self.data = self.list_frames 
        elif cfg.size < 1:
            np.random.shuffle(self.list_frames)
            self.data = self.list_frames[:int(cfg.size * self.list_frames.__len__())]
        else:
            np.random.shuffle(self.list_frames)
            self.data = self.list_frames[:cfg.size]
        # ! By default this dataloader iterates by frames
        logging.info(f"Simple MLC dataloader initialized with: {self.cfg.data_dir.img_dir}")
        logging.info(f"Labels reading from: {os.path.join(self.cfg.data_dir.labels_dir, self.cfg.label)}")
        logging.info(f"Total number of frames:{self.data.__len__()}")
        
    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx
        filename = os.path.splitext(self.data[idx])[0]
        label_fn = os.path.join(self.cfg.data_dir.labels_dir, self.cfg.label, f"{filename}")
            
        std_fn = os.path.join(self.cfg.data_dir.labels_dir, 'std', f"{filename}.npy")
        image_fn = os.path.join(self.cfg.data_dir.img_dir, f"{filename}")
        
        if os.path.exists(image_fn + '.jpg'):
            image_fn += '.jpg'
        elif os.path.exists(image_fn + '.png'):
            image_fn += '.png'
        
        img = np.array(Image.open(image_fn), np.float32)[..., :3] / 255.
        
        if os.path.exists(label_fn + ".npy"):
            label = np.load(label_fn + ".npy")
        elif os.path.exists(label_fn + ".npz"):
            label = np.load(label_fn + ".npz")["phi_coords"]
        else:
            raise ValueError(f"Not found {label_fn}")
        
        # Random flip
        if self.cfg.get('flip', False) and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=len(label.shape) - 1)

        # Random horizontal rotate
        if self.cfg.get('rotate', False):
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            label = np.roll(label, dx, axis=len(label.shape) - 1)

        # Random gamma augmentation
        if self.cfg.get('gamma', False):
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img**p
        
        if os.path.exists(std_fn):
            std = np.load(std_fn)
        else:
            std = np.ones_like(label)
        
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        std = torch.FloatTensor(std.copy())
        return [x, label, std]

