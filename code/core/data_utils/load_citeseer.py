import numpy as np
import torch
import random
import os
import csv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from core.utils import one_hot_embedding
from core.config import sim_Add, sim_Delete

def get_citeseer_gpt_pred():
    result_list = []
    with open('gpt_preds/citeseer.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if ',' in row:
                num = row[0]
                result_list.append(int(num[0]))
            else:
                result_list.append(int(row[0]))
    return result_list

def get_citeseer_casestudy(SEED=0):
    processed_data = torch.load('dataset/citeseer/citeseer_fixed_sbert.pt')
    data_X, data_Y, data_edges = processed_data.x, processed_data.y, processed_data.edge_index.numpy().T
    set_list = []

    for row in data_edges:
        row_set = set()
        row_set.add(row[0])
        row_set.add(row[1])
        set_list.append(row_set)
    
    unique_list = []
    for row_set in set_list:
        if row_set not in unique_list:
            unique_list.append(row_set)
    data_edges = np.array([list(x) for x in unique_list])

    thr_add = sim_Add
    thr_delete = sim_Delete
    ori_data_edges = data_edges
    final_deleted_edges = torch.load('add_delete/add_delete_citeseer/edges_plus_delete_' + str(thr_delete) + '_0_1000.pt')
    final_added_edges = torch.load('add_delete/add_delete_citeseer/edges_plus_add_' + str(thr_add) + '_0_1000.pt')
    A = final_added_edges
    B = ori_data_edges
    diff_indices = np.setdiff1d(np.arange(A.shape[0]), np.where((A[:, None] == B).all(axis=2))[0])
    diff = A[diff_indices]
    data_edges = np.vstack((final_deleted_edges, diff))

    data_edges = np.unique(np.vstack((data_edges, np.fliplr(data_edges))), axis=0)
    data_edges = np.unique(data_edges, axis=0).transpose()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    data_name = 'citeseer'
    dataset = Planetoid('dataset', data_name,
                        transform = T.ToSparseTensor())
    data = dataset[0]
    
    data.x = processed_data.x
    data.num_nodes = len(processed_data.y)
    data.y = torch.tensor(processed_data.y).long()
    data.edge_index = torch.tensor(data_edges).long()
    data.edge_index = SparseTensor(row = data.edge_index[1], col = data.edge_index[0], sparse_sizes=(data.num_nodes, data.num_nodes))

    data.labels_for_lpa = one_hot_embedding(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    data.gpt_pred = torch.tensor(get_citeseer_gpt_pred()).long()
    data.labels_for_lpa_gpt = one_hot_embedding(data.gpt_pred, data.gpt_pred.max().item() + 1).type(torch.FloatTensor)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data

def get_raw_text_citeseer(use_text=False, seed=0):
    data = get_citeseer_casestudy(seed)
    if not use_text:
        return data, None
    
    text = torch.load('dataset/citeseer/citeseer_fixed_sbert.pt').raw_texts
    return data, text
