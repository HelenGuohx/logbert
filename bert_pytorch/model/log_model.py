import torch.nn as nn
import torch
from .bert import BERT

class BERTLog(nn.Module):
    """
    BERT Log Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        self.time_lm = TimeLogModel(self.bert.hidden)
        # self.fnn_cls = LinearCLS(self.bert.hidden)
        #self.cls_lm = LogClassifier(self.bert.hidden)
        self.result = {"logkey_output": None, "time_output": None, "cls_output": None, "cls_fnn_output": None}

    def forward(self, x, time_info):
        x = self.bert(x, time_info=time_info)

        self.result["logkey_output"] = self.mask_lm(x)
        # self.result["time_output"] = self.time_lm(x)

        # self.result["cls_output"] = x.float().mean(axis=1) #x[:, 0]
        self.result["cls_output"] = x[:, 0]
        # self.result["cls_output"] = self.fnn_cls(x[:, 0])

        # print(self.result["cls_fnn_output"].shape)

        return self.result

class MaskedLogModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class TimeLogModel(nn.Module):
    def __init__(self, hidden, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidden, time_size)

    def forward(self, x):
        return self.linear(x)

class LogClassifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, cls):
        return self.linear(cls)

class LinearCLS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)