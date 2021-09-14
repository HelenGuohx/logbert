import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLog, BERT
from .optim_schedule import ScheduledOptim
import time
import tqdm
import numpy as np
import pandas as pd

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, is_logkey=True, is_time=False,
                 hypersphere_loss=False):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param valid_dataloader: valid dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLog(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and valid data loader
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optim = None
        self.optim_schedule = None
        self.init_optimizer()


        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.time_criterion = nn.MSELoss()
        self.hyper_criterion = nn.MSELoss()

        # deep SVDD hyperparameters
        self.hypersphere_loss = hypersphere_loss
        self.radius = 0
        self.hyper_center = None
        self.nu = 0.25
        # self.objective = "soft-boundary"
        self.objective = None

        self.log_freq = log_freq

        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.is_logkey = is_logkey
        self.is_time = is_time

    def init_optimizer(self):
        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=self.warmup_steps)

    def train(self, epoch):
        return self.iteration(epoch, self.train_data, start_train=True)

    def valid(self, epoch):
        return self.iteration(epoch, self.valid_data, start_train=False)

    def iteration(self, epoch, data_loader, start_train):
        """
        loop over the data_loader for training or validing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or valid
        :return: None
        """
        str_code = "train" if start_train else "valid"

        lr = self.optim.state_dict()['param_groups'][0]['lr']
        start = time.strftime("%H:%M:%S")
        self.log[str_code]['lr'].append(lr)
        self.log[str_code]['time'].append(start)

        # Setting the tqdm progress bar
        totol_length = len(data_loader)
        # data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
        data_iter = enumerate(data_loader)

        total_loss = 0.0
        total_logkey_loss = 0.0
        total_hyper_loss = 0.0

        total_dist = []
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            result = self.model.forward(data["bert_input"], data["time_input"])
            mask_lm_output, mask_time_output = result["logkey_output"], result["time_output"]

            # 2-2. NLLLoss of predicting masked token word ignore_index = 0 to ignore unmasked tokens
            mask_loss = torch.tensor(0) if not self.is_logkey else self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            total_logkey_loss += mask_loss.item()

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss

            # hypersphere loss
            if self.hypersphere_loss:
                # version 1.0
                # hyper_loss = self.hyper_criterion(result["cls_fnn_output"].squeeze(), self.hyper_center.expand(data["bert_input"].shape[0],-1))
                hyper_loss = self.hyper_criterion(result["cls_output"].squeeze(), self.hyper_center.expand(data["bert_input"].shape[0], -1))

                # version 2.0 https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
                dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
                total_dist += dist.cpu().tolist()

                # if self.objective == 'soft-boundary':
                #     scores = dist - self.radius ** 2
                #     hyper_loss = torch.sqrt(self.radius ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)))
                # else:
                #     hyper_loss = torch.sqrt(torch.mean(dist))

                # # add radius and center to training
                # self.radius = self.get_radius(dist, self.nu)
                # self.hyper_center = torch.mean(result["cls_output"], dim=0)

                total_hyper_loss += hyper_loss.item()

                # with deepsvdd loss
                loss = loss + 0.1 * hyper_loss

            total_loss += loss.item()

            # 3. backward and optimization only in train
            if start_train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

        avg_loss = total_loss / totol_length
        self.log[str_code]['epoch'].append(epoch)
        self.log[str_code]['loss'].append(avg_loss)
        print("Epoch: {} | phase: {}, loss={}".format(epoch, str_code, avg_loss))
        print(f"logkey loss: {total_logkey_loss/totol_length}, hyper loss: {total_hyper_loss/totol_length}\n")

        return avg_loss, total_dist

    def save_log(self, save_dir, surfix_log):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(save_dir + key + f"_{surfix_log}.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def save(self, save_dir="output/bert_trained.pth"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        torch.save(self.model, save_dir)
        # self.bert.to(self.device)
        print(" Model Saved on:", save_dir)
        return save_dir

    @staticmethod
    def get_radius(dist: list, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist), 1 - nu)


