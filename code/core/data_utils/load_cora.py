import numpy as np
import torch
import random
import os
import csv

from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from core.utils import one_hot_embedding
from core.config import sim_Add, sim_Delete

def get_cora_gpt_pred():
    result_list = []
    with open('gpt_preds/cora.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if ',' in row:
                num = row[0]
                result_list.append(int(num[0]))
            else:
                result_list.append(int(row[0]))
    return result_list

def get_cora_gcn_pred(SEED=0):
    result_list = []
    with open('gcn_preds/cora_' + str(SEED) + '.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if ',' in row:
                num = row[0]
                result_list.append(int(num[0]))
            else:
                result_list.append(int(row[0]))
    return result_list

def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.\
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    data_name = 'cora'
    dataset = Planetoid('dataset', data_name,
                        transform=T.ToSparseTensor())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.edge_index = SparseTensor(row = data.edge_index[1], col = data.edge_index[0])

    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    data.labels_for_lpa = one_hot_embedding(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    data.gpt_pred = torch.tensor(get_cora_gpt_pred()).long()
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

    return data, data_citeid

def parse_cora():
    path = 'dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    
    pname = 'add_delete_cora'
    thr_add = sim_Add
    thr_delete = sim_Delete
    ori_data_edges = data_edges
    final_deleted_edges = torch.load('add_delete/'+pname+'/edges_plus_delete_' + str(thr_delete) + '_0_1000.pt')
    final_added_edges = torch.load('add_delete/'+pname+'/edges_plus_add_' + str(thr_add) + '_0_1000.pt')
    A = final_added_edges
    B = ori_data_edges
    diff_indices = np.setdiff1d(np.arange(A.shape[0]), np.where((A[:, None] == B).all(axis=2))[0])
    diff = A[diff_indices]
    data_edges = np.vstack((final_deleted_edges, diff))
    
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('dataset/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'dataset/cora_orig/mccallum/cora/extractions/'
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)
    return data, text