import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling


def return_unk():
    return UNK


def get_length(x):
    # 这里axis=-1是取最后一维进行求和，      这里的sum(1)==sum(axis=1)
    #          2199 500 20->2199 50->2199
    # 每一条数据的500里面的前几百行都是0
    # 这里求得的是500行里面的有效数据是多少行
    return x.shape[1]-(np.sum(x, axis=-1) == 0).sum(1)


class MOSI:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # load pickle file for unaligned acoustic and visual source
            # pickle_filename = '../datasets/MOSI/mosi_data_noalign.pkl'
            # csv_filename = '../datasets/MOSI/MOSI-label.csv'

            pickle_filename = 'Multimodal-Infomax-main/datasets/MOSI/mosi_data_noalign.pkl'
            csv_filename = 'Multimodal-Infomax-main/datasets/MOSI/MOSI-label.csv'

            with open(pickle_filename, 'rb') as f:
                # 函数的功能：将file中的对象序列化读出。什么叫序列化？
                # 把对象在内存中的结构转换成便于存储或传输的二进制或文本格式，
                # 而且以后可以在同一个系统或不同的系统中重建对象的副本。
                # pickle模块能把任何Python对象序列化成二进制格式。
                d = pickle.load(f)

            # read csv file for label and text   004 :1 cid_id
            # vid 000'03bSnISJMiM'001:'03bSnISJMiM'002:'03bSnISJMiM'003:'03bSnISJMiM'004:'03bSnISJMiM'
            # text 001:'THERE IS SAD PART'
            df = pd.read_csv(csv_filename)
            text = df['text']  # 2199个   获得所有的text 数据
            vid = df['video_id']  # 2199 获得所有数据的video_id
            cid = df['clip_id']  # 2199  获得所有数据的 clip_id
            # 在这里猜测每一条数据有一个id label vision text audio
            # 这个id由两部分组成一部分是video另一部分是clip_id
            train_split_noalign = d['train']  # 1284
            dev_split_noalign = d['valid']  # 229
            test_split_noalign = d['test']  # 686

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)_(.*)')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            if True:
                v = np.concatenate(
                    (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
                vlens = get_length(v)

                a = np.concatenate(
                    (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
                alens = get_length(a)
                # 这里的label和从csv里面读取出来label是一样的
                label = np.concatenate(
                    (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)
                # label[label > 0] = 2
                # label[label < 0] = -2
                print(label.sort())
                L_V = v.shape[1]
                L_A = a.shape[1]

            all_id = np.concatenate(
                (train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            # 这个是把all_id的编码方式换成了utf-8，原来编码是’S14‘
            all_id_list = list(
                map(lambda x: x.decode('utf-8'), all_id.tolist()))

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset
                # 把idd分成标号和字符根据  '_'
                idd1, idd2 = re.search(pattern, idd).group(1, 2)

                # matching process
                try:
                    index = all_csv_id.index((idd1, idd2))
                except:
                    exit()
                """
                    Retrive noalign data from pickle file 
                """
                # 这里的text是文字，而经过d = pickle.load(f)读取出来的text不是文字，所以直接使用。
                # 应该是这样，要使得文字与另外两个模态对齐
                _words = text[index].split()
                _label = label[i].astype(np.float32)  # 这条数据对应的标签
                _visual = v[i]  # 视觉
                _acoustic = a[i]  # 声音
                _vlen = vlens[i]  # 视觉长度
                _alen = alens[i]  # 声音长度
                _id = all_id[i]

                # remove nan values 默认是用0代替nan的值
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                # For non-align setting
                # we also need to record sequence lengths
                """TODO: Add length counting for other datasets 
                """
                for word in _words:
                    actual_words.append(word)

                # 单个数据的前几百行都是0，数据都在后面，为什么在后面不清楚
                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                # z-normalization per instance and remove nan/infs
                # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                # 这里是把配对的数据组成一个集合放到一起
                if i < dev_start:
                    train.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(
                        f"Found video that doesn't belong to any splits: {idd}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):
        if mode == "train":
            # 这里的self.train和train引用一个对象所以用self.train也是可以的
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/trainsss.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # first we align to words with averaging, collapse_function receives a list of functions
            # dataset.align(text_field, collapse_functions=[avg])
            # load pickle file for unaligned acoustic and visual source
            pickle_filename = 'Multimodal-Infomax-main/datasets/MOSEI/mosei_senti_data_noalign.pkl'
            csv_filename = 'Multimodal-Infomax-main/datasets/MOSEI/MOSEI-label.csv'

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']

            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            # pattern = re.compile('(.*)\[.*\]')
            pattern = re.compile('(.*)_([.*])')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            v = np.concatenate(
                (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
            vlens = get_length(v)

            a = np.concatenate(
                (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
            alens = get_length(a)

            label = np.concatenate(
                (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

            L_V = v.shape[1]
            L_A = a.shape[1]

            all_id = np.concatenate(
                (train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']), axis=0)[:, 0]
            all_id_list = all_id.tolist()

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset

                # matching process
                try:
                    index = i
                except:
                    import ipdb
                    ipdb.set_trace()

                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = '{}[{}]'.format(all_csv_id[0], all_csv_id[1])

                # remove nan values
                # label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                for word in _words:
                    actual_words.append(word)

                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                if i < dev_start:
                    train.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(
                        ((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(
                        f"Found video that doesn't belong to any splits: {idd}")

            # print(f"Total number of {num_drop} datapoints have been dropped.")
            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)
            self.pretrained_emb = None

            # Save pickles保存数据
            to_pickle(train, DATA_PATH + '/trainsss.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()
