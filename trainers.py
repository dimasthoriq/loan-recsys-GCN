"""
Author: Dimas Ahmad
Description: This file contains the training function for the project.
"""

import torch
import time
import os
from models import STHCW
from utils import build_adjacency, symmetric_normalization, to_torch_sparse, split_train_test
from evals import bpr_loss, ndcg_at_k, precision_at_k, recall_at_k


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, best_loss, current_loss):
        if current_loss <= (best_loss - self.min_delta):
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    def __init__(self, R, P, Q, config):
        self.R = R
        self.P = P
        self.Q = Q
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n, self.m = R.shape
        self.p = P.shape[1]
        self.q = Q.shape[1]

        A = build_adjacency(R, P, Q, weighted=config['weighted'])
        A = symmetric_normalization(A)
        self.A = to_torch_sparse(A).to(self.device)

        self.train_edges, self.test_edges = split_train_test(R, train_size=config['train_size'])
        self.train_investors = torch.LongTensor(self.train_edges[:, 0]).to(self.device)
        self.train_loans = torch.LongTensor(self.train_edges[:, 1]).to(self.device)
        self.test_investors = torch.LongTensor(self.test_edges[:, 0]).to(self.device)
        self.test_loans = torch.LongTensor(self.test_edges[:, 1]).to(self.device)

        self.model = STHCW(
            n=self.n,
            m=self.m,
            p=self.p,
            q=self.q,
            k=config['layers'],
            embedding_dim=config['embed_dim'],
            weighted=config['weighted']
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config['learning_rate'],
                                          weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    factor=config['sched_factor'],
                                                                    patience=config['sched_patience'],
                                                                    )
        self.early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])

        self.start_time = time.time()
        self.best_model = self.model.state_dict()
        self.best_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        neg_loans = torch.randint(0, self.m, (len(self.train_investors),)).to(self.device)

        e = self.model(self.A)
        investor_e = e[self.train_investors]
        pos_loan_e = e[self.n+self.train_loans]
        neg_loan_e = e[self.n+neg_loans]

        pos_scores = torch.sum(investor_e * pos_loan_e, dim=1)
        neg_scores = torch.sum(investor_e * neg_loan_e, dim=1)
        loss = bpr_loss(pos_scores, neg_scores)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            neg_loans = torch.randint(0, self.m, (len(self.test_investors),)).to(self.device)

            e = self.model(self.A)
            investor_e = e[self.test_investors]
            pos_loan_e = e[self.n+self.test_loans]
            neg_loan_e = e[self.n+neg_loans]

            pos_scores = torch.sum(investor_e * pos_loan_e, dim=1)
            neg_scores = torch.sum(investor_e * neg_loan_e, dim=1)
            loss = bpr_loss(pos_scores, neg_scores)
        return loss.item()

    def validate(self, investors, loans):
        self.model.eval()
        with torch.no_grad():
            e = self.model(self.A)

            ndgc_25_list = []
            precision_25_list = []
            recall_25_list = []
            ndgc_50_list = []
            precision_50_list = []
            recall_50_list = []

            for investor in torch.unique(investors):
                pos_loan = loans[investors == investor]
                if pos_loan.numel() == 0:
                    continue

                investor_e = e[investor]
                all_loan_e = e[self.n: self.n+self.m]

                scores = torch.matmul(all_loan_e, investor_e)
                target = torch.zeros_like(self.m).to(self.device)
                target[pos_loan] = 1

                ndgc_25 = ndcg_at_k(target, scores, 25)
                precision_25 = precision_at_k(target, scores, 25)
                recall_25 = recall_at_k(target, scores, 25)

                ndgc_50 = ndcg_at_k(target, scores, 50)
                precision_50 = precision_at_k(target, scores, 50)
                recall_50 = recall_at_k(target, scores, 50)

                ndgc_25_list.append(ndgc_25)
                precision_25_list.append(precision_25)
                recall_25_list.append(recall_25)
                ndgc_50_list.append(ndgc_50)
                precision_50_list.append(precision_50)
                recall_50_list.append(recall_50)

            avg_ndgc_25 = torch.mean(torch.tensor(ndgc_25_list))
            avg_precision_25 = torch.mean(torch.tensor(precision_25_list))
            avg_recall_25 = torch.mean(torch.tensor(recall_25_list))

            avg_ndgc_50 = torch.mean(torch.tensor(ndgc_50_list))
            avg_precision_50 = torch.mean(torch.tensor(precision_50_list))
            avg_recall_50 = torch.mean(torch.tensor(recall_50_list))
            return avg_ndgc_25, avg_precision_25, avg_recall_25, avg_ndgc_50, avg_precision_50, avg_recall_50

    def train(self):
        train_losses = []
        val_losses = []

        for epoch in range(self.config['epochs']):
            epoch_train_loss = self.train_epoch()
            epoch_val_loss = self.test_epoch()

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            if (epoch + 1) % self.config['val_every'] == 0:
                ndgc25, pr25, rc25, ndgc50, pr50, rc50 = self.validate(self.test_investors,
                                                                       self.test_loans)
                print(f'Epoch {epoch + 1}, NDCG@25: {ndgc25:.4f}, '
                      f'Precision@25: {pr25:.4f}, Recall@25: {rc25:.4f}, '
                      f'NDCG@50: {ndgc50:.4f}, Precision@50: {pr50:.4f}, Recall@50: {rc50:.4f}')

            if (epoch + 1) % self.config['print_every'] == 0:
                print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, '
                      f'Val Loss: {epoch_val_loss:.4f}, '
                      f'Best: {self.best_epoch+1}')

            self.scheduler.step(epoch_val_loss)
            self.early_stopping(self.best_loss, epoch_val_loss)

            if self.early_stopping.early_stop:
                print('Early stopping')
                print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, '
                      f'Val Loss: {epoch_val_loss:.4f}, '
                      f'Best: {self.best_epoch + 1}')
                break

            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss
                self.best_epoch = epoch
                self.best_model = self.model.state_dict()

        duration = time.time() - self.start_time
        print('Training completed in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        self.model.load_state_dict(self.best_model)

        save_dir = self.config['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        stamp = time.strftime('%Y%m%d_%H%M', time.localtime())
        save_path = os.path.join(save_dir, f'model_{stamp}.pth')
        torch.save(self.model, save_path)
        return train_losses, val_losses
