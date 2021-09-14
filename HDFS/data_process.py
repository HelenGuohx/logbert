import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from logparser import Spell, Drain

# get [log key, delta time] as input for deeplog
input_dir  = os.path.expanduser('~/.dataset/hdfs/')
output_dir = '../output/hdfs/'  # The output directory of parsing results
log_file   = "HDFS.log"  # The input log file name

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"

def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(input_dir, output_dir, log_file, log_format, type='drain'):
    if type == 'spell':
        tau        = 0.5  # Message type threshold (default: 0.5)
        regex      = [
            "(/[-\w]+)+", #replace file path with *
            "(?<=blk_)[-\d]+" #replace block_id with *

        ]  # Regular expression list for optional preprocessing (default: [])

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
        parser.parse(log_file)

    elif type == 'drain':
        regex = [
            r"(?<=blk_)[-\d]+", # block_id
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r"(/[-\w]+)+",  # file path
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 5  # Depth of all leaf nodes


        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        parser.parse(log_file)


def hdfs_sampling(log_file, window='session'):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    with open(output_dir + "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict = defaultdict(list) #preserve insertion order of items
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict[blk_Id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv(log_sequence_file, index=None)
    print("hdfs sampling done")


def generate_train_test(hdfs_sequence_file, n=None, ratio=0.3):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(hdfs_sequence_file)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    df_to_file(train, output_dir + "train")
    df_to_file(test_normal, output_dir + "test_normal")
    df_to_file(test_abnormal, output_dir + "test_abnormal")
    print("generate train test data done")


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


if __name__ == "__main__":
    # 1. parse HDFS log
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    # parser(input_dir, output_dir, log_file, log_format, 'drain')
    # mapping()
    # hdfs_sampling(log_structured_file)
    generate_train_test(log_sequence_file, n=4855)
