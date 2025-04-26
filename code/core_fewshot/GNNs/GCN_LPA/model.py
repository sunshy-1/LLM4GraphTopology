import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_sparse import SparseTensor,spmm
from core_fewshot.GNNs.GCN_LPA.layers import GCNConv

from torch_scatter import gather_csr, segment_csr
from torch_sparse import SparseTensor

def softmax(src: SparseTensor, dim=1):
    value = src.storage.value()
    rowptr = src.storage.rowptr()

    value_exp = torch.exp(value)
    sum_value_exp = segment_csr(value_exp, rowptr)
    sum_value_exp = gather_csr(sum_value_exp, rowptr)

    return src.set_value(value_exp / sum_value_exp, layout='csr')

class GCN_LPA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred, adj, num_lpa_iter, num_edges, device):
        super(GCN_LPA, self).__init__()
        self.use_pred = use_pred
        self.num_lpa_iter = num_lpa_iter

        self.num_edges = num_edges
        self.edge_weight = nn.Parameter(torch.ones(self.num_edges))
        self.device = device
        self.dropout = dropout
        self.num_classes = out_channels

        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels+1, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=True, use_norm = True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=True, use_norm = True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=True, use_norm = True))
        
        self.convs_lpa = torch.nn.ModuleList()
        lpa_in_channels = self.num_classes
        for _ in range(self.num_lpa_iter):
            self.convs_lpa.append(GCNConv(lpa_in_channels, lpa_in_channels, cached=False, bias=False, not_use_lpa=False, add_self_loops=True, use_norm = True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj, labels_for_lpa, labels_for_lpa_gpt):
        self.adj_ori = adj.set_value(self.edge_weight)
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, self.adj_ori)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, self.adj_ori)
        
        y_hat = labels_for_lpa.reshape(-1,self.num_classes)
        for i, conv in enumerate(self.convs_lpa):
            y_hat = conv(y_hat, self.adj_ori)
        
        y_hat2 = labels_for_lpa_gpt.reshape(-1,self.num_classes)
        for i, conv in enumerate(self.convs_lpa):
            y_hat2 = conv(y_hat2, self.adj_ori)

        return x, y_hat, y_hat2