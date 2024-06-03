import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import MMIM


class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        # hyp_params==args
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        
        self.model = model
        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.nce2 = hp.nce2
        self.nce3 = hp.nce3
        self.y_true = torch.tensor([])
        self.y_pre = torch.tensor([])
        self.p_value =[]
        self.t_statistic = []

        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = MMIM(hp)

        if torch.cuda.is_available():
            # if False:
            self.device = torch.device("cuda:0")
            # self.device = torch.device("cuda:7")
            model = model.to(self.device)

        else:
            self.device = torch.device("cpu")

        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []
            
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else:
                        main_param.append(p)
                    
                for p in (mmilb_param+main_param):
                    if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        # nn.init.xavier_normal_()
                        nn.init.xavier_normal_(p)

        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )
        

        self.scheduler_main = ReduceLROnPlateau(
            self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model

        optimizer_main = self.optimizer_main

        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = 1

        def train(model, optimizer, criterion,epochs, stage=1):
            epoch_loss = 0
            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            ba_loss = 0.0
            start_time = time.time()

            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                model.zero_grad()
                dd = torch.device("cuda:0")
                # dd = torch.device("cuda:7")
                with torch.cuda.device(0):
                    # with torch.cuda.device(3):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                        text.to(dd), visual.to(dd), audio.to(dd), y.to(dd), l.to(dd), bert_sent.to(dd), \
                        bert_sent_type.to(dd), bert_sent_mask.to(dd)

                    if self.hp.dataset == "ur_funny":
                        y = y.squeeze()

                batch_size = y.size(0)

                # visiual==(261，32，20) audio==(218,32,5)
                nce, preds, nce2, nce3 = model(text, visual, audio, vlens, alens,
                                               bert_sent, bert_sent_type, bert_sent_mask, y)
                if stage == 1:
                    y_loss = criterion(preds, y)

                    if self.hp.contrast:
                        loss = y_loss + self.alpha * nce+self.nce2*nce2+self.nce3 * nce3
                    else:
                        loss = y_loss
                    loss.backward()

                else:
                    raise ValueError('stage index can either be 0 or 1')

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(  
                        model.parameters(), self.hp.clip)
                    optimizer.step()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                nce_loss += nce.item() * batch_size

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size
                    avg_ba = ba_loss / proc_size
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                          format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, 'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
                                 avg_loss, avg_nce, avg_ba))
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, epochs,test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0

            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                    dd = torch.device("cuda:0")
                    with torch.cuda.device(0):
                        text, audio, vision, y = text.to(
                            dd), audio.to(dd), vision.to(dd), y.to(dd)
                        lengths = lengths.to(dd)
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(dd
                                                                                 ), bert_sent_type.to(dd), bert_sent_mask.to(dd)
                        if self.hp.dataset == 'iemocap':
                            y = y.long()

                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    # bert_sent in size (bs, seq_len, emb_size)
                    batch_size = lengths.size(0)

                    # we don't need lld and bound anymore
                    _, preds, _, _ = model(
                        text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)

            avg_loss = total_loss / \
                (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths
            
        patience = self.hp.patience
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            train_loss = train(model, optimizer_main, criterion, epoch,1)
            val_loss, _, _ = evaluate(model, criterion,epoch, test=False)
            test_loss, results, truths = evaluate(model, criterion,epoch, test=True)

            end = time.time()
            duration = end-start
            # Decay learning rate by validation loss
            scheduler_main.step(val_loss)

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
                epoch, duration, val_loss, test_loss))
            print("-"*50)

            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        eval_mosei_senti(results, truths, True)

                    elif self.hp.dataset == 'mosi':
                        eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_iemocap(results, truths)

                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models/MM.pt!")
                    save_model(self.hp, model)
            else:
                patience -= 1
                if patience == 0:
                    break
        
        torch.save(self.p_value,"mosi_mcl_mcf_p_value.pth")
        torch.save(self.t_statistic,"mosi_mcl_mcf_t_statistic.pth")

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)
        sys.stdout.flush()
