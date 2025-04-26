import torch
import os
import csv
import random

import pandas as pd
import numpy as np

from torch_sparse import SparseTensor
from core.utils import one_hot_embedding
from core.config import sim_Add, sim_Delete

def get_arxiv_2023_gpt_pred():
    result_list = []
    with open('gpt_preds/arxiv_2023.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                result_list.append(int(1))
                continue
            if ',' in row:
                num = row[0]
                result_list.append(int(num[0]))
            else:
                result_list.append(int(row[0]))
    return result_list


def get_raw_text_arxiv_2023(use_text=False, seed=0):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = torch.load('dataset/arxiv_2023/graph.pt')

    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    data_edges = torch.load('add_delete/add_delete_arxiv_2023/data_edges_arxiv_2023.pt')

    thr_add = sim_Add
    thr_delete = sim_Delete

    if sim_Add > 1 and sim_Delete < 0:
        data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
        data.edge_index = torch.tensor(np.unique(data_edges, axis=0).transpose())
    else:
        ori_data_edges = data_edges
        final_deleted_edges = torch.load('add_delete/add_delete_arxiv_2023/edges_plus_delete_' + str(thr_delete) + '_0_1000.pt')
        final_added_edges = torch.load('add_delete/add_delete_arxiv_2023/edges_plus_add_' + str(thr_add) + '_0_1000.pt')
        A = final_added_edges
        B = ori_data_edges
    
        dict_diff = {}
        A_minus_B = []
        for idxA in range(len(A)):
            keyA = tuple(A[idxA])
            if keyA in dict_diff:
                dict_diff[keyA] += 1
            else:
                dict_diff[keyA] = 1
            
        for idxB in range(len(B)):
            keyB = tuple(B[idxB])
            if keyB in dict_diff:
                dict_diff[keyB] -= 1
            
        for key_dict_diff in dict_diff.keys():
            for _ in range(dict_diff[key_dict_diff]):
                A_minus_B.append(key_dict_diff) 
        A_minus_B = np.array(A_minus_B)
        diff = A_minus_B
    
        if len(diff) == 0:
            data_edges = final_deleted_edges
        else:
            data_edges = np.vstack((final_deleted_edges, diff))
    
        data_edges = np.unique(np.vstack((data_edges, np.fliplr(data_edges))), axis=0)
        data_edges = np.unique(data_edges, axis=0).transpose()
        data.edge_index = torch.tensor(data_edges)

    data.edge_index = SparseTensor(row = data.edge_index[1], col = data.edge_index[0], sparse_sizes=(data.num_nodes, data.num_nodes))

    data.labels_for_lpa = one_hot_embedding(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    data.gpt_pred = torch.tensor(get_arxiv_2023_gpt_pred()).long()
    data.labels_for_lpa_gpt = one_hot_embedding(data.gpt_pred, data.gpt_pred.max().item() + 1).type(torch.FloatTensor)

    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    if not use_text:
        return data, None

    df = pd.read_csv('dataset/arxiv_2023_orig/paper_info.csv')
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
    return data, text
