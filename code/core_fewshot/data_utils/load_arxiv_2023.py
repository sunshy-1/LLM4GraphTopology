import torch
import os
import csv
import random

import pandas as pd
import numpy as np

from torch_sparse import SparseTensor
from core_fewshot.utils import one_hot_embedding, get_few_shot_samples
from core_fewshot.config import sim_Add, sim_Delete

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
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    data = torch.load('dataset/arxiv_2023/graph.pt')

    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    
    data_edges = torch.load('add_delete/add_delete_arxiv_2023/data_edges_arxiv_2023.pt')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    data.edge_index = torch.tensor(np.unique(data_edges, axis=0).transpose())
    data.edge_index = SparseTensor(row = data.edge_index[1], col = data.edge_index[0], sparse_sizes=(data.num_nodes, data.num_nodes))

    data.labels_for_lpa = one_hot_embedding(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    data.gpt_pred = torch.tensor(get_arxiv_2023_gpt_pred()).long()
    data.labels_for_lpa_gpt = one_hot_embedding(data.gpt_pred, data.gpt_pred.max().item() + 1).type(torch.FloatTensor)

    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    few_shot = True
    if few_shot:
        data.train_id, data.val_id, data.test_id = get_few_shot_samples(data, node_id, numTrain_perclass = 20, numVal=500, numTest=1000)

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
    return data, text
