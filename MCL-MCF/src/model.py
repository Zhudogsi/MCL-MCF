import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, Clip, get_resNet, get_resNet2, get_resNet3, get_resNet4

from transformers import BertModel, BertConfig
from modules.transformer import TransformerEncoder
import torchvision
from torchvision import models


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        # 'if add va MMILB module---add_va'
        self.add_va = hp.add_va
        # args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim  768 20 5
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        # 2. Crossmodal Attentions

        self.visual_enc = RNNEncoder(  # (rnn): LSTM(20, 16) Linear(in_features=16, out_features=16, bias=True)
            in_size=hp.d_vin,  # 20
            hidden_size=hp.d_vh,  # 16
            out_size=hp.d_vout,  # 16
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(  # (rnn): LSTM(20, 16)Linear(in_features=16, out_features=16, bias=True)
            in_size=hp.d_ain,  # 5
            hidden_size=hp.d_ah,  # 16
            out_size=hp.d_aout,  # 16
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        self.fusion_prj = SubNet(  # (linear_1): Linear(in_features=800, out_features=128, bias=True)
            # (linear_2): Linear(in_features=128, out_features=128, bias=True)
            # (linear_3): Linear(in_features=128, out_features=1, bias=True)
            in_size=512,
            hidden_size=hp.d_prjh,  # 128
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.embed_dim = hp.embed_dim
        self.num_heads = hp.num_heads
        self.layers = hp.layers
        self.attn_dropout = hp.attn_dropout
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout
        self.embed_dropout = hp.embed_dropout
        self.attn_mask = hp.attn_mask

        self.ta_clip = Clip(768, 64)
        self.tv_clip = Clip(768, 64)
        self.av_clip = Clip(64, 64)
        self.mass_t_clip = Clip(768, 64)
        self.mass_v_clip = Clip(64, 64)
        self.mass_a_clip = Clip(64, 64)
        self.mass_tav_clip = Clip(64, 64)

        self.conv_tv = nn.Conv1d(
            in_channels=832, out_channels=64, kernel_size=1)
        self.conv_av = nn.Conv1d(
            in_channels=128, out_channels=64, kernel_size=1)
        self.conv_vt = nn.Conv1d(
            in_channels=832, out_channels=64, kernel_size=1)
        self.conv_tav = nn.Conv1d(
            in_channels=896, out_channels=64, kernel_size=1)
        self.conv_res = nn.Conv1d(
            in_channels=2048, out_channels=512, kernel_size=1)
        self.getresNet = get_resNet()
        self.getresNet2 = get_resNet2()
        self.getresNet3 = get_resNet3()
        self.getresNet4 = get_resNet4()
        self.tfn_tv = Clip(256036, 256036)  
        self.tfn_ta = Clip(256036, 256036)
        self.tfn_av = Clip(256036, 256036)

        # resnet50 = models.resnet50(pretrained=True).add_module(
        #     nn.Flatten())  

        self.conv_tfn_t = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4,  kernel_size=(3, 3)),

            nn.ReLU(),
            nn.Conv2d(
                in_channels=4, out_channels=6,  kernel_size=(3, 3)),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=(3, 3)), nn.ReLU(),
            nn.Flatten()

        )
        self.conv_tfn_v = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4,  kernel_size=(3, 3)), nn.ReLU(),

            nn.Conv2d(
                in_channels=4, out_channels=6,   kernel_size=(3, 3)), nn.ReLU(),

            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=(3, 3)), nn.ReLU(),
            nn.Flatten()
        )
        self.conv_tfn_a = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4,  kernel_size=(3, 3)), nn.ReLU(),

            nn.Conv2d(
                in_channels=4, out_channels=6,   kernel_size=(3, 3)),  nn.ReLU(),

            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=(3, 3)), nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """  # (32,50,768)
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type,
                                 bert_sent_mask)  # (batch_size, seq_len, emb_size)->(32,50,768)
        
        text = enc_word[:, 0, :]  # (batch_size, emb_size)32,768
        
        acoustic, aco_rnn_output = self.acoustic_enc(
            acoustic, a_len)  # (218,32,5)-> (32,64) #aco_rnn_output->[32, 64, 218])
        visual, vis_rnn_output = self.visual_enc(
            visual, v_len)  # (261,32,20)-> (32,64) vis_rnn_output->[261, 32, 16])  49152

        massagehub_tv = torch.cat([text, visual], dim=1)  # 32,832
        massagehub_va = torch.cat([visual, acoustic], dim=1)  # 32,128
        massagehub_ta = torch.cat([text, acoustic], dim=1)  # 32,832
        massagehub_tav = torch.cat([text, visual, acoustic], dim=1)
        massagehub_tv = self.conv_tv(
            massagehub_tv.unsqueeze(2)).squeeze(2)  # 32,64
        massagehub_va = self.conv_av(massagehub_va.unsqueeze(2)).squeeze(2)
        massagehub_ta = self.conv_vt(massagehub_ta.unsqueeze(2)).squeeze(2)
        massagehub_tav = self.conv_tav(massagehub_tav.unsqueeze(2)).squeeze(2)

        # fusion, preds = self.fusion_prj(
        #     torch.cat([text, acoustic, visual], dim=1))  # 32,896)
        tav = torch.cat([text, acoustic, visual], dim=1).unsqueeze(2)  # 32,896
        qwer = self.getresNet4(tav).squeeze(2)

        res = self.getresNet(text.unsqueeze(2)).squeeze(2)  # 32 512
        res2 = self.getresNet2(visual.unsqueeze(2)).squeeze(2)  # 32 512
        res3 = self.getresNet3(acoustic.unsqueeze(2)).squeeze(2)  # 32 512

        tfn_text_visual = torch.bmm(
            res.unsqueeze(2), res2.unsqueeze(1)).unsqueeze(1)  # [32, 1,512, 512]
        tfn_text_acoustic = torch.bmm(
            res.unsqueeze(2), res3.unsqueeze(1)).unsqueeze(1)  # [32,1, 512, 512]
        tfn_visual_acoustic = torch.bmm(
            res2.unsqueeze(2), res3.unsqueeze(1)).unsqueeze(1)  # [32, 1,512, 512]
        tfn_text_visual = self.conv_tfn_t(
            tfn_text_visual)  # [32, 16386304]  [32, 2048288] 32, 2000000 32, 1952288 1016064
        tfn_text_acoustic = self.conv_tfn_v(
            tfn_text_acoustic)
        tfn_visual_acoustic = self.conv_tfn_a(
            tfn_visual_acoustic)

        res_all = torch.cat([res, res2, res3], dim=1)  # [32, 1536]
        xxx = torch.cat([res_all, qwer], dim=1).unsqueeze(2)
        res_mm = self.conv_res(xxx).squeeze(2)

        fusion, preds = self.fusion_prj(res_mm)  # 32, 512, 1

        clip_ta = self.ta_clip(text, acoustic)
        clip_tv = self.tv_clip(text, visual)
        clip_av = self.av_clip(visual, acoustic)

        clip_mass_t = self.mass_t_clip(text, massagehub_va)
        clip_mass_v = self.mass_v_clip(visual, massagehub_ta)
        clip_mass_a = self.mass_a_clip(acoustic, massagehub_tv)

        clip_mass_tav = self.mass_tav_clip(acoustic, massagehub_tav)

        tfn_tv_loss = self.tfn_tv(tfn_text_visual, tfn_text_acoustic)
        tfn_ta_loss = self.tfn_ta(tfn_text_visual, tfn_visual_acoustic)
        tfn_av_loss = self.tfn_av(tfn_text_acoustic, tfn_visual_acoustic)

        # tfn_tav_loss = self.tfn_tav(
        #     tfn_text_visual_acoustic, tfn_text_visual_acoustic)

        nce = clip_ta+clip_tv + clip_av
        nce2 = clip_mass_t+clip_mass_v+clip_mass_a+clip_mass_tav

        nce3 = tfn_tv_loss+tfn_ta_loss+tfn_av_loss

        return nce, preds, nce2, nce3
