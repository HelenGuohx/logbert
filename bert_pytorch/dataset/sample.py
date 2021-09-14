from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than 10
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])

    return logkey_seqs, time_seq


def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs)
    time_seq_pairs = np.array(time_seq_pairs)

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(logkey_seq_pairs,
                                                                                      time_seq_pairs,
                                                                                      test_size=test_size,
                                                                                      random_state=1234)

    # sort seq_pairs by seq len
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset

