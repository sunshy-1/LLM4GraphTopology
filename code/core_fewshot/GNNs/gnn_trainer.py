import torch
from time import time
import numpy as np

from core_fewshot.GNNs.GCN.model import GCN
from core_fewshot.GNNs.GCN_LPA.model import GCN_LPA
from core_fewshot.GNNs.gnn_utils import EarlyStopping
from core_fewshot.data_utils.load import load_data, load_gpt_preds, load_gcn_preds
from core_fewshot.utils import time_logger

LOG_FREQ = 10


class GNNTrainer():

    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs
        self.num_lpa_iter = cfg.gnn.train.num_lpa_iter
        self.lamda_lpa = cfg.gnn.train.lamda_lpa
        self.lamda_lpa_2 = cfg.gnn.train.lamda_lpa_2

        data = load_data(self.dataset_name, use_dgl=False,
                         use_text=False, seed=self.seed)

        self.num_nodes = data.x.shape[0]
        self.num_classes = data.y.max().item() + 1 if self.dataset_name == 'arxiv_2023' else data.y.unique().size(0) 
        self.num_edges = data.edge_index.nnz()
        
        data.y = data.y.squeeze()

        if self.lm_model_name == 'microsoft/deberta-base':
            print('deberta-base')
            feat_size = 768
        else:
            print('e5-large')
            feat_size = 1024

        topk = 3 if self.dataset_name == 'pubmed' else 5
        
        if self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"prt_lm_fewshot/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, feat_size)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"prt_lm_fewshot/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, feat_size)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        else:
            print(
                f'Feature type {self.feature_type} not supported.')

        self.features = features.to(self.device)
        self.data = data.to(self.device)
        self.labels_for_lpa = data.labels_for_lpa.to(self.device)
        self.labels_for_lpa_gpt = data.labels_for_lpa_gpt.to(self.device)
        self.labels_for_lpa[self.data.train_mask == False] = 0

        use_pred = self.feature_type == 'P'
        if self.gnn_model_name == "GCN":
            self.model = GCN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                             hidden_channels=self.hidden_dim,
                             out_channels=self.num_classes,
                             num_layers=self.num_layers,
                             dropout=self.dropout,
                             use_pred=use_pred).to(self.device)

        elif self.gnn_model_name == 'GCN_LPA':
            self.model = GCN_LPA(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                             hidden_channels=self.hidden_dim,
                             out_channels=self.num_classes,
                             num_layers=self.num_layers,
                             dropout=self.dropout,
                             use_pred=use_pred,
                             adj = self.data.edge_index,
                             num_lpa_iter = self.num_lpa_iter,
                             num_edges = self.num_edges,
                             device = self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.001)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core_fewshot.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        return logits
    
    def GCN_LPA_forward(self, x, edge_index, labels_for_lpa, labels_for_lpa_gpt):
        logits, y_hat, y_hat2 = self.model(x, edge_index, labels_for_lpa, labels_for_lpa_gpt)
        return logits, y_hat, y_hat2

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        if self.gnn_model_name == "GCN_LPA":
            logits, y_hat, y_hat2 = self.GCN_LPA_forward(self.features, self.data.edge_index, self.labels_for_lpa, self.labels_for_lpa_gpt)
            
            loss_gcn = self.loss_func(
                logits[self.data.train_mask], self.data.y[self.data.train_mask])
            loss_lpa = self.loss_func(
                y_hat[self.data.train_mask], self.data.y[self.data.train_mask])
            loss_lpa2 = self.loss_func(
                y_hat2[self.data.train_mask], self.data.y[self.data.train_mask])
            
            loss = loss_gcn + self.lamda_lpa * loss_lpa + self.lamda_lpa_2 * loss_lpa2

            train_acc = self.evaluator(
                logits[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            
            return loss.item(), loss_gcn.item(), loss_lpa.item(), loss_lpa2.item(), train_acc

        logits = self._forward(self.features, self.data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        if self.gnn_model_name == "GCN_LPA":
            logits, _, _ = self.GCN_LPA_forward(self.features, self.data.edge_index, self.labels_for_lpa, self.labels_for_lpa_gpt)
        else:
            logits = self._forward(self.features, self.data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            if self.gnn_model_name == "GCN_LPA":
                loss, loss_gcn, loss_lpa, loss_lpa2, train_acc = self._train()
            else:
                loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                if self.gnn_model_name == "GCN_LPA":
                    print(f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, Loss_gcn: {loss_gcn:.4f}, Loss_lpa:{loss_lpa:.4f}, Loss_lpa2:{loss_lpa2:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')
                else:
                    print(f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        print(
            f'[{self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res
