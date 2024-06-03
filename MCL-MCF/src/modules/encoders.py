import torch
import torch.nn.functional as F
import time

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np

MODEL_PATH = 'src/bert/' 
def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens


class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """

    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = BertConfig.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH, output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained(
             pretrained_model_name_or_path=MODEL_PATH, config=bertconfig)

    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)
        # print(self.bertmodel,"zheshi bert，Odell")
        bert_output = bert_output[0]
        return bert_output   # return head (sequence representation)

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)  # 192   128
        self.linear_2 = nn.Linear(hidden_size, hidden_size)  # 128   128
        self.linear_3 = nn.Linear(hidden_size, n_class)  # 128    1

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_2, y_3


class CLUB(nn.Module):
    """
        Compute the Contrastive Log-ratio Upper Bound (CLUB) given a pair of inputs.
        Refer to https://arxiv.org/pdf/2006.12013.pdf and https://github.com/Linear95/CLUB/blob/f3457fc250a5773a6c476d79cda8cb07e1621313/MI_DA/MNISTModel_DANN.py#L233-254

        Args:
            hidden_size(int): embedding size
            activation(int): the activation function in the middle layer of MLP
    """

    def __init__(self, hidden_size, activation='Tanh'):
        super(CLUB, self).__init__()
        try:
            self.activation = getattr(nn, activation)
        except:
            raise ValueError(
                "Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, modal_a, modal_b, sample=False):
        """
            CLUB with random shuffle, the Q function in original paper:
                CLUB = E_p(x,y)[log q(y|x)]-E_p(x)p(y)[log q(y|x)]

            Args:
                modal_a (Tensor): x in above equation
                model_b (Tensor): y in above equation
        """
        mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(
            modal_a)  # (bs, hidden_size)
        batch_size = mu.size(0)
        pred = mu

        # pred b using a
        pred_tile = mu.unsqueeze(1).repeat(
            1, batch_size, 1)   # (bs, bs, emb_size)
        true_tile = pred.unsqueeze(0).repeat(
            batch_size, 1, 1)      # (bs, bs, emb_size)

        positive = - (mu - modal_b) ** 2 / 2. / torch.exp(logvar)
        negative = - torch.mean((true_tile-pred_tile) **
                                2, dim=1)/2./torch.exp(logvar)

        lld = torch.mean(torch.sum(positive, -1))
        bound = torch.mean(torch.sum(positive, -1)-torch.sum(negative, -1))
        return lld, bound


class MMILB(nn.Module):
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """

    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:  
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError(
                "Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            # 768  16
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            # 16   16
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            # 768   16
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            # 16    16
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            # 16    4
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        # print(self.mlp_mu)
        # print(self.mlp_logvar)
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x)  # (bs, hidden_size)
        batch_size = mu.size(0)
        # slist = []
        # for x in logvar:
        #     n_list = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        #     n_list = torch.where(torch.isinf(
        #         n_list), torch.full_like(n_list, 0), n_list)
        #     n_list[n_list > 88] = 88
        #     n_list = torch.exp(n_list)
        #     n_list = torch.where(torch.isinf(
        #         n_list), torch.full_like(n_list, 0), n_list)
        #     slist.append(n_list)

        # s_sum = torch.stack(slist, dim=0)
        positive = -(mu - y)**2/2./torch.exp(logvar)

        # positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive, -1))

        # For Gaussian Distribution Estimation
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos': None, 'neg': None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y)

            pos_y = y[labels.squeeze() > 0]
            neg_y = y[labels.squeeze() < 0]

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # Diagonal setting
                # pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                # neg_all = torch.cat(neg_history + [neg_y], dim=0)
                # mu_pos = pos_all.mean(dim=0)
                # mu_neg = neg_all.mean(dim=0)

                # sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
                # sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
                # H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

                # compute the entire co-variance matrix
                pos_all = torch.cat(pos_history + [pos_y], dim=0)  # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm(
                    (pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
                sigma_neg = torch.mean(torch.bmm(
                    (neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
                a = 17.0795
                H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))

        return lld, sample_dict, H

#from Learning Transferable Visual Models From Natural Language Supervision (https://github.com/OpenAI/CLIP)
class Clip(nn.Module):
    def __init__(self, x_size, y_size):
        super().__init__()

        self.x_learn_weight = nn.Linear(x_size, 128)
        self.y_learn_weight = nn.Linear(y_size, 128)

    def forward(self, x, y):  # 32,768  32,64
        x = self.x_learn_weight(x)  # 32,128
        logit_scale = torch.ones([]) * np.log(1 / 0.07)
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        y = self.y_learn_weight(y)
        logits = logit_scale*torch.mm(x, y.T)
        labels = torch.arange(x.shape[0])
        # logits[logits > 88] = 88
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i+loss_t)/2.


class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calulate the score 
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)    # bs, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)  # norm各元素平方求和开根号
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''                 20    16           16
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        
        self.bidirectional = bidirectional
#                           20            16         1
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(
            (2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.to(torch.int64)
        bs = x.size(0)
        
        lengths = lengths.to(torch.int64)
        packed_sequence = pack_padded_sequence(  
            x, lengths.cpu().to(torch.int64), enforce_sorted=False)
        rnn_output, final_states = self.rnn(packed_sequence)  # 输入压缩
        encoder_outputs, lens = pad_packed_sequence(
            rnn_output)  

        if self.bidirectional:
            h = self.dropout(
                torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:
            # h.shape==32*16
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1, encoder_outputs


# @@@@@@@@@@@@@@@@
class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def get_resNet():
    b1 = nn.Sequential(nn.Conv1d(768, 512, kernel_size=1, stride=1),
                       nn.ReLU())
    # b2 = nn.Sequential(*resnet_block(512, 256, 1),
    #                    nn.ReLU())
    # b3 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())
    # b4 = nn.Sequential(*resnet_block(128, 256, 1),
    #                    nn.ReLU())
    # b5 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())

    net = nn.Sequential(b1, )
    return net


def get_resNet2():
    b1 = nn.Sequential(nn.Conv1d(64, 512, kernel_size=1, stride=1),
                       nn.ReLU())
    # b2 = nn.Sequential(*resnet_block(512, 256, 1),
    #                    nn.ReLU())
    # b3 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())
    # b4 = nn.Sequential(*resnet_block(128, 256, 1),
    #    nn.ReLU())
    # b5 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())

    net = nn.Sequential(b1, )
    return net


def get_resNet3():
    b1 = nn.Sequential(nn.Conv1d(64, 512, kernel_size=1, stride=1),
                       nn.ReLU())
    # b2 = nn.Sequential(*resnet_block(512, 256, 1),
    #                    nn.ReLU())
    # b3 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())
    # b4 = nn.Sequential(*resnet_block(128, 256, 1),
    #                    nn.ReLU())
    # b5 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())

    net = nn.Sequential(b1, )
    return net


def get_resNet4():
    b1 = nn.Sequential(nn.Conv1d(896, 512, kernel_size=1, stride=1),
                       nn.ReLU())
    # b2 = nn.Sequential(*resnet_block(512, 256, 1),
    #                    nn.ReLU())
    # b3 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())
    # b4 = nn.Sequential(*resnet_block(128, 256, 1),
    #                    nn.ReLU())
    # b5 = nn.Sequential(*resnet_block(256, 128, 1),
    #                    nn.ReLU())

    net = nn.Sequential(b1, )
    return net
