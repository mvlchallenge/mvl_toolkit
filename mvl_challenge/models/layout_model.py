import os
from mvl_challenge.utils.io_utils import save_json_dict
from mvl_challenge.config.cfg import save_cfg
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np


class LayoutModel:
    def __init__(self, cfg):
        self.cfg = cfg        
    
    def train_loop(self):
        pass 

    def valid_loop(self):
        pass
    
    def prepare_for_training(self):
        self.is_training = True
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()
        self.set_optimizer()
        self.set_scheduler()
        self.set_train_dataloader()
        self.set_log_dir()
        save_cfg(os.path.join(self.dir_ckpt, 'cfg.yaml'), self.cfg)

    def set_log_dir(self):
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        logging.info(f"Output directory: {output_dir}")
        self.dir_log = os.path.join(output_dir, 'log')
        self.dir_ckpt = os.path.join(output_dir, 'ckpt')
        os.makedirs(self.dir_log, exist_ok=True)
        os.makedirs(self.dir_ckpt, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.dir_log)

    def set_train_dataloader(self):
        logging.info("Setting Training Dataloader")
        pass
    
    
    def set_valid_dataloader(self):
        logging.info("Setting IoU Validation Dataloader")
        pass
    
    def set_optimizer(self):
        if self.cfg.model.optimizer == "SGD":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                momentum=self.cfg.model.beta1,
                weight_decay=self.cfg.model.weight_decay,
            )
        elif self.cfg.model.optimizer == "Adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                betas=(self.cfg.model.beta1, 0.999),
                weight_decay=self.cfg.model.weight_decay,
            )
        else:
            raise NotImplementedError()

    def set_scheduler(self):
        decayRate = self.cfg.model.lr_decay_rate
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )
