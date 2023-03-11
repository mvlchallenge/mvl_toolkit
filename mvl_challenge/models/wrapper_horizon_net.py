
import json
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from copy import deepcopy
from mvl_challenge.config.cfg import save_cfg
from mvl_challenge import ROOT_DIR
from mvl_challenge.datasets.mvl_dataset import MVImageLayout


class WrapperHorizonNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_horizon_net_path()
        from mvl_challenge.models.HorizonNet.dataset import visualize_a_data
        from mvl_challenge.models.HorizonNet.misc import utils as hn_utils
        from mvl_challenge.models.HorizonNet.model import HorizonNet

        # ! Setting cuda-device
        self.device = torch.device(
            f"cuda:{cfg.cuda}" if torch.cuda.is_available() else 'cpu')

        # Loaded trained model
        assert os.path.isfile(cfg.model.ckpt), "Not found {cfg.model.ckpt}"
        logging.info("Loading HorizonNet...")
        self.net = hn_utils.load_trained_model(
            HorizonNet, cfg.model.ckpt).to(self.device)
        logging.info(f"ckpt: {cfg.model.ckpt}")
        logging.info("HorizonNet Wrapper Successfully initialized")

    @staticmethod
    def set_horizon_net_path():
        hn_dir = os.path.join(ROOT_DIR, "models", "HorizonNet")
        if hn_dir not in sys.path:
            sys.path.append(hn_dir)

    def estimate_within_list_ly(self, list_ly):
        """
        Estimates phi_coords (layout boundaries) for all ly defined in list_ly using the passed model instance
        """

        layout_dataloader = DataLoader(
            MVImageLayout([(ly.img_fn, ly.idx) for ly in list_ly]),
            batch_size=self.cfg.runners.mvl.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.mvl.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(),
        )
        self.net.eval()
        evaluated_data = {}
        for x in tqdm(layout_dataloader, desc=f"Estimating layout..."):
            with torch.no_grad():
                y_bon_, y_cor_ = self.net(x['images'].to(self.device))
                # y_bon_, y_cor_ = net(x[0].to(device))
            for y_, cor_, idx in zip(y_bon_.cpu(), y_cor_.cpu(), x['idx']):
                data = np.vstack((y_, cor_))
                evaluated_data[idx] = data

        [ly.recompute_data(phi_coords=evaluated_data[ly.idx]) for ly in list_ly]
