import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import MOSI, MOSEI, PAD, UNK
MODEL_PATH = '/media/data2/zhukang/goodjob/goodjob/new_7_2____2_2_-2good-1aaa2/Multimodal-Infomax-main/src/bert/' # 装着上面3个文件的文件夹位置
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH,do_lower_case=True)



class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        # Fetch dataset
        # data_dir==/media/data2/zhukang/first_pro/Multimodal-Infomax-main/datasets
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        else:
            print("Dataset not defined correctly")
            exit()
        # 返回对齐的数据
        self.data, self.word2id, _ = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.word2id = self.word2id
        # config.pretrained_emb = self.pretrained_emb

    @property  # 这个的作用是可以把tva_dim当作属性使用
    def tva_dim(self):
        t_dim = 768
        # print(self.data[0][0][1], "wwwwwwwwwwwwwwwwwwww",
        #       self.data[0][0][1].shape[1])
        # print(self.data[0][0][2], "wwwwwwwwwwwwwwwwwwww",
        #       self.data[0][0][2].shape[1])
        # 这里是  768   vision的一条数据的长度20       audio的一条数据的长度 5
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print(config.mode)
    config.data_len = len(dataset)
    # 返回 t  v  a的长度
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    # 这里的batch_size是由DataLoader构造的batch_size,
    # collate_fn对这个batch_szie进行处理然后返回真正的batch_szie--real min_batch_szie

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''  # 32*
        # for later use we sort the batch in descending order of length
        # x[0][3]这个是text的内容，所以这个sorted是按照text的字长来排序倒序，大的在前面
        # 这对后面的  pad_sequence 有用
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)
        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            if len(sample[0]) > 4:  # unaligned case 作者这里可能注释写错了，这是对齐了的数据
                v_lens.append(torch.IntTensor([sample[0][4]]))  # vision的有效长度
                a_lens.append(torch.IntTensor([sample[0][5]]))  # audio的有效长度
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            labels.append(torch.from_numpy(sample[1]))  # sample[1]是label
            ids.append(sample[2])  # 一串英文字母后面的那个数字Oz06ZWiO20M_11  11就是idd作用还不清楚
        # 把一个batch的每条数据的vision的有效长度并起来，共32个 torch.Size([32])
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)  # 同上torch.Size([32])
        labels = torch.cat(labels, dim=0)  # torch.Size([32, 1]) 是个二维的

        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]

        # Rewrite this
        # 对这个函数的详细解释https://blog.csdn.net/wangchaoxjtu/article/details/118023187
        # https://blog.csdn.net/devil_son1234/article/details/108660887
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:  # 这个会生成一个元组->(max_len, len(sequences),trailing_dims)
                out_dims = (max_len, len(sequences)) + trailing_dims
            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:  # ...就是好几个冒号
                    out_tensor[i, :length, ...] = tensor
                else:
                    # (261,32,20)visual它是中间维度32每次只用一个维度，把他们都填充成了261，32，20的矩阵
                    # 例如一个数据是（80，20）那么选取261行的前80行作为真实数据填充，后面的是0.0数据填充
                    # 所以是填充成为最长的数据，对于音频则是填充成为长度是218的数据
                    out_tensor[:length, i, ...] = tensor
            return out_tensor
        a = [torch.LongTensor(sample[0][0])for sample in batch]
        b = [torch.FloatTensor(sample[0][1])for sample in batch]
        c = [torch.FloatTensor(sample[0][2])for sample in batch]
        sentences = pad_sequence([torch.LongTensor(sample[0][0])
                                  for sample in batch], padding_value=PAD)

        visual = pad_sequence([torch.FloatTensor(sample[0][1])
                               for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2])
                                 for sample in batch], target_len=alens.max().item())

        # BERT-based features input prep

        # SENT_LEN = min(sentences.size(0),50)
        SENT_LEN = 50
        # Create bert indices using tokenizer
        # 使用预训练好的bert对文本进行填充和编码，长度是50
        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        # Bert things are batch_first 把编译好的文本的三个东西，分别取出来
        bert_sentences = torch.LongTensor(
            [sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor(
            [sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor(
            [sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            # 把vlens中为0的部分赋值为1
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids
    # 构建了一个dateloader batch_size 默认是32
    data_loader = DataLoader(
        # 这里的dataset=MSADataset重写了这个方法
        #  def __getitem__(self, index):return self.data[index]
        # 所以dataset里面有很多东西，但是在DataLoader进行的batch_size数据的构造会盗用__getitem__函数来构造
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,  # 打乱
        collate_fn=collate_fn,  # 解释https://www.iotword.com/3015.html
        generator=torch.Generator(device='cuda'))  # 使用CPU

    return data_loader
