import sys
sys.path.append("../")
sys.path.append("../../")

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../deeplog')


import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from logdeep.tools.utils import *

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "../output/tbird/"
options["model_dir"] = options["output_dir"] + "bert/"
options["train_vocab"] = options["output_dir"] + "train"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"

options["model_path"] = options["model_dir"] + "best_bert.pth"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10

options["mask_ratio"] = 0.5

options["vocab_size"] = 844

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 0.1


# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 15
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)
print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        Trainer(options).train()

    elif args.mode == 'predict':
        Predictor(options).predict()

    elif args.mode == 'vocab':
        with open(options["train_vocab"], "r") as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])





