# adapted from https://github.com/jcatw/scnn
import torch
import random
import json
import os
import csv
import pandas as pd
import numpy as np
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import normalize
from torch_sparse import SparseTensor
from core.utils import one_hot_embedding
from core.config import sim_Add, sim_Delete


# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs
def get_pubmed_gpt_pred():
    result_list = []
    with open('gpt_preds/pubmed.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if ',' in row:
                num = row[0]
                result_list.append(int(num[0]))
            else:
                result_list.append(int(row[0]))
    return result_list

def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.edge_index = SparseTensor(row = data.edge_index[1], col = data.edge_index[0], sparse_sizes=(data.num_nodes, data.num_nodes))
    data.y = torch.tensor(data_Y)
    data.num_nodes = len(data.y)

    data.labels_for_lpa = one_hot_embedding(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    data.gpt_pred = torch.tensor(get_pubmed_gpt_pred()).long()
    data.labels_for_lpa_gpt = one_hot_embedding(data.gpt_pred, data.gpt_pred.max().item() + 1).type(torch.FloatTensor)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        data.train_id = [i for i in data.train_id if not is_mistake[i]]
        data.val_id = [i for i in data.val_id if not is_mistake[i]]
        data.test_id = [i for i in data.test_id if not is_mistake[i]]

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    
    return data, data_pubid


def parse_pubmed():
    path = 'dataset/PubMed_orig/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                # data_edges.append(
                #     (paper_to_index[tail], paper_to_index[head]))
        data_edges = np.array(data_edges)
        
        thr_add = sim_Add
        thr_delete = sim_Delete
        ori_data_edges = data_edges
        final_deleted_edges = torch.load('add_delete/add_delete_pubmed/edges_plus_delete_' + str(thr_delete) + '_0_1000.pt')
        final_added_edges = torch.load('add_delete/add_delete_pubmed/edges_plus_add_' + str(thr_add) + '_0_1000.pt')
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

    return data_A, data_X, data_Y, data_pubid, data_edges


def get_raw_text_pubmed(use_text=False, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
    if not use_text:
        return data, None

    f = open('dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)
    return data, text
