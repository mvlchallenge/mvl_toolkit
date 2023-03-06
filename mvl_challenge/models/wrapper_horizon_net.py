
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
from mvl_challenge.models.layout_model import LayoutModel
from mvl_challenge import ROOT_DIR
# from mvl_challenge.data_loaders.mlc_mix_dataloader import MLC_MixedDataDataLoader
# from mvl_challenge.data_loaders.mlc_simple_dataloader import (ListImgLayout, ListLayout,
#                                                     MLC_SimpleDataLoader)
# from mvl_challenge.utils.mvl_utils import load_mvl_dataset, load_mvl_data_with_multithread
# from mvl_challenge.utils.entropy_mcl_utils import eval_entropy_from_boundaries, eval_entropy_from_bounds_hist
# from mvl_challenge.utils.info_utils import get_mean_mse_h, print_run_information
# from mvl_challenge.utils.io_utils import save_json_dict
# from mvl_challenge.utils.layout_utils import filter_out_noisy_layouts
# from mvl_challenge.utils.loss_and_eval_utils import *
# from mvl_challenge.utils.entropy_mcl_utils import normalize_list_ly, estimate_h_with_multithread


class WrapperHorizonNet(LayoutModel):
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
