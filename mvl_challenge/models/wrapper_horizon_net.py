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
from mvl_challenge.utils.io_utils import save_json_dict, print_cfg_information, create_directory
from mvl_challenge.data_loaders.mvl_dataloader import MVLDataLoader
from mvl_challenge.utils.eval_utils import eval_2d3d_iuo_from_tensors, compute_weighted_L1, compute_L1_loss


class WrapperHorizonNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_horizon_net_path()
        from mvl_challenge.models.HorizonNet.misc import utils as hn_utils
        from mvl_challenge.models.HorizonNet.model import HorizonNet

        # ! Setting cuda-device
        self.device = torch.device(
            f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else "cpu"
        )

        # Loaded trained model
        assert os.path.isfile(cfg.model.ckpt), f"Not found {cfg.model.ckpt}"
        logging.info("Loading HorizonNet...")
        self.net = hn_utils.load_trained_model(HorizonNet, cfg.model.ckpt).to(
            self.device
        )
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
            pin_memory=True if self.device != "cpu" else False,
            worker_init_fn=lambda x: np.random.seed(),
        )
        self.net.eval()
        evaluated_data = {}
        for x in tqdm(layout_dataloader, desc=f"Estimating layout..."):
            with torch.no_grad():
                y_bon_, y_cor_ = self.net(x["images"].to(self.device))
                # y_bon_, y_cor_ = net(x[0].to(device))
            for y_, cor_, idx in zip(y_bon_.cpu(), y_cor_.cpu(), x["idx"]):
                data = np.vstack((y_, cor_))
                evaluated_data[idx] = data
        [ly.set_phi_coords(phi_coords=evaluated_data[ly.idx]) for ly in list_ly]

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

    def train_loop(self):
        if not self.is_training:
            logging.warning("Wrapper is not ready for training")
            return False

        # ! Freezing some layer
        if self.cfg.model.freeze_earlier_blocks != -1:
            b0, b1, b2, b3, b4 = self.net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(self.cfg.model.freeze_earlier_blocks + 1):
                logging.warn('Freeze block %d' % i)
                for m in blocks[i]:
                    for param in m.parameters():
                        param.requires_grad = False

        if self.cfg.model.bn_momentum != 0:
            for m in self.net.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = self.cfg.model.bn_momentum

        print_cfg_information(self.cfg)

        self.net.train()

        iterator_train = iter(self.train_loader)
        for _ in trange(len(self.train_loader),
                        desc=f"Training HorizonNet epoch:{self.current_epoch}/{self.cfg.model.epochs}"):

            self.iterations += 1
            x, y_bon_ref, std = next(iterator_train)
            y_bon_est, _ = self.net(x.to(self.device))

            if y_bon_est is np.nan:
                raise ValueError("Nan value")

            if self.cfg.model.loss == "L1":
                loss = compute_L1_loss(y_bon_est.to(
                    self.device), y_bon_ref.to(self.device))
            elif self.cfg.model.loss == "weighted_L1":
                loss = compute_weighted_L1(y_bon_est.to(
                    self.device), y_bon_ref.to(self.device), std.to(self.device), self.cfg.model.min_std)
            else:
                raise ValueError("Loss function no defined in config file")
            if loss.item() is np.NAN:
                raise ValueError("something is wrong")
            self.tb_writer.add_scalar(
                "train/loss", loss.item(), self.iterations)
            self.tb_writer.add_scalar(
                "train/lr", self.lr_scheduler.get_last_lr()[0], self.iterations)

            # back-prop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), 3.0, norm_type="inf")
            self.optimizer.step()

        self.lr_scheduler.step()

        # Epoch finished
        self.current_epoch += 1

        # ! Saving model
        if self.cfg.model.get("save_every") > 0:
            if self.current_epoch % self.cfg.model.get("save_every", 5) == 0:
                self.save_model(f"model_at_{self.current_epoch}.pth")

        if self.current_epoch > self.cfg.model.epochs:
            self.is_training = False

        # # ! Saving current epoch data
        # fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        # save_json_dict(filename=fn, dict_data=self.curr_scores)

        return self.is_training

    def save_current_scores(self):
        # ! Saving current epoch data
        fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        save_json_dict(filename=fn, dict_data=self.curr_scores)
        # ! Save the best scores in a json file regardless of saving the model or not
        save_json_dict(
            dict_data=self.best_scores,
            filename=os.path.join(self.dir_ckpt, "best_score.json")
        )

    def valid_iou_loop(self, only_val=False):
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        total_eval = {}
        invalid_cnt = 0

        for _ in trange(len(iterator_valid_iou), desc="IoU Validation epoch %d" % self.current_epoch):
            x, y_bon_ref, std = next(iterator_valid_iou)

            with torch.no_grad():
                y_bon_est, _ = self.net(x.to(self.device))

                true_eval = {"2DIoU": [], "3DIoU": []}
                for gt, est in zip(y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy()):
                    eval_2d3d_iuo_from_tensors(est[None], gt[None], true_eval, )


                local_eval = dict(
                    loss=compute_weighted_L1(y_bon_est.to(
                        self.device), y_bon_ref.to(self.device), std.to(self.device))
                )
                local_eval["2DIoU"] = torch.FloatTensor(
                    [true_eval["2DIoU"]]).mean()
                local_eval["3DIoU"] = torch.FloatTensor(
                    [true_eval["3DIoU"]]).mean()
            try:
                for k, v in local_eval.items():
                    if v.isnan():
                        continue
                    total_eval[k] = total_eval.get(k, 0) + v.item() * x.size(0)
            except:
                invalid_cnt += 1
                pass

        if only_val:
            scaler_value = self.cfg.runners.valid_iou.batch_size * \
                (len(iterator_valid_iou) - invalid_cnt)
            curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
            curr_score_2d_iou = total_eval["2DIoU"] / scaler_value
            logging.info(f"3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"2D-IoU score: {curr_score_2d_iou:.4f}")
            return {"2D-IoU": curr_score_2d_iou, "3D-IoU": curr_score_3d_iou}

        scaler_value = self.cfg.runners.valid_iou.batch_size * \
            (len(iterator_valid_iou) - invalid_cnt)
        for k, v in total_eval.items():
            k = "valid_IoU/%s" % k
            self.tb_writer.add_scalar(
                k, v / scaler_value, self.current_epoch)

        # Save best validation loss model
        curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
        curr_score_2d_iou = total_eval["2DIoU"] / scaler_value

        # ! Saving current score
        self.curr_scores['iou_valid_scores'] = dict(
            best_3d_iou_score=curr_score_3d_iou,
            best_2d_iou_score=curr_score_2d_iou
        )

        if self.best_scores.get("best_iou_valid_score") is None:
            logging.info(f"Best 3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"Best 2D-IoU score: {curr_score_2d_iou:.4f}")
            self.best_scores["best_iou_valid_score"] = dict(
                best_3d_iou_score=curr_score_3d_iou,
                best_2d_iou_score=curr_score_2d_iou
            )
        else:
            best_3d_iou_score = self.best_scores["best_iou_valid_score"]['best_3d_iou_score']
            best_2d_iou_score = self.best_scores["best_iou_valid_score"]['best_2d_iou_score']

            logging.info(
                f"3D-IoU: Best: {best_3d_iou_score:.4f} vs Curr:{curr_score_3d_iou:.4f}")
            logging.info(
                f"2D-IoU: Best: {best_2d_iou_score:.4f} vs Curr:{curr_score_2d_iou:.4f}")

            if best_3d_iou_score < curr_score_3d_iou:
                logging.info(
                    f"New 3D-IoU Best Score {curr_score_3d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_3d_iou_score'] = curr_score_3d_iou
                self.save_model("best_3d_iou_valid.pth")

            if best_2d_iou_score < curr_score_2d_iou:
                logging.info(
                    f"New 2D-IoU Best Score {curr_score_2d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_2d_iou_score'] = curr_score_2d_iou
                self.save_model("best_2d_iou_valid.pth")

    def save_model(self, filename):
        if not self.cfg.model.get("save_ckpt", True):
            return

        # ! Saving the current model
        state_dict = OrderedDict(
            {
                "args": self.cfg,
                "kwargs": {
                    "backbone": self.net.backbone,
                    "use_rnn": self.net.use_rnn,
                },
                "state_dict": self.net.state_dict(),
            }
        )
        torch.save(state_dict, os.path.join(
            self.dir_ckpt, filename))

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
        create_directory(output_dir, delete_prev=False)
        logging.info(f"Output directory: {output_dir}")
        self.dir_log = os.path.join(output_dir, 'log')
        self.dir_ckpt = os.path.join(output_dir, 'ckpt')
        os.makedirs(self.dir_log, exist_ok=True)
        os.makedirs(self.dir_ckpt, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.dir_log)

    def set_train_dataloader(self):
        logging.info("Setting Training Dataloader")
        self.train_loader = DataLoader(
            MVLDataLoader(self.cfg.runners.train),
            batch_size=self.cfg.runners.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.runners.train.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(),
        )

    def set_valid_dataloader(self):
        logging.info("Setting IoU Validation Dataloader")
        self.valid_iou_loader = DataLoader(
            MVLDataLoader(self.cfg.runners.valid_iou),
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())
