# cora llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core_fewshot.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset cora \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 2.0 \

# citeseer llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core_fewshot.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset citeseer \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 5.8 \

# pubmed llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core_fewshot.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset pubmed \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 2.0 \

# arxiv_2023 llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core_fewshot.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset arxiv_2023 \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 2.0 \