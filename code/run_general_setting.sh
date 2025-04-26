# cora GCN
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset cora \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 \
    gnn.train.lamda_lpa 1.1 gnn.train.lamda_lpa_2 3.0 gnn.train.num_lpa_iter 3 \
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# cora GCN + llm-based A-D
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset cora \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 \
    gnn.train.lamda_lpa 1.1 gnn.train.lamda_lpa_2 3.0 gnn.train.num_lpa_iter 3 \
    gnn.train.sim_Add 0.7 gnn.train.sim_Delete 0.2

# cora GCN + llm-bsed LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset cora \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 \
    gnn.train.lamda_lpa 1.1 gnn.train.lamda_lpa_2 3.0 gnn.train.num_lpa_iter 3 \
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# cora GCN + llm-based A-D & LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset cora \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 \
    gnn.train.lamda_lpa 1.1 gnn.train.lamda_lpa_2 3.0 gnn.train.num_lpa_iter 3 \
    gnn.train.sim_Add 0.7 gnn.train.sim_Delete 0.2

# ---------------------------------------------------------------------------------------------------------------------------#
# citeseer GCN + llm-based A-D
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset citeseer \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.5 gnn.train.lamda_lpa_2 0.1\
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# citeseer GCN + llm-based A-D
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset citeseer \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.5 gnn.train.lamda_lpa_2 0.1\
    gnn.train.sim_Add 0.2 gnn.train.sim_Delete 0.1

# citeseer GCN + llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset citeseer \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.MIX False gnn.train.retrain_gcn False \
    gnn.train.lamda_lpa 0.5 gnn.train.lamda_lpa_2 0.8 \
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# citeseer GCN + llm-based A-D & LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset citeseer \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.5 gnn.train.lamda_lpa_2 0.1\
    gnn.train.sim_Add 0.2 gnn.train.sim_Delete 0.1

# ---------------------------------------------------------------------------------------------------------------------------#
# pubmed GCN
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset pubmed \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.1 gnn.train.lamda_lpa_2 2.2\
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# pubmed GCN + llm-based A-D
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset pubmed \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.1 gnn.train.lamda_lpa_2 2.2\
    gnn.train.sim_Add 0.9 gnn.train.sim_Delete 0.4

# pubmed GCN + llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset pubmed \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.1 gnn.train.lamda_lpa_2 2.2\
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# pubmed GCN + llm-based A-D & LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset pubmed \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 1 \
    gnn.train.lamda_lpa 0.1 gnn.train.lamda_lpa_2 2.2\
    gnn.train.sim_Add 0.9 gnn.train.sim_Delete 0.3

# ---------------------------------------------------------------------------------------------------------------------------#
# arxiv_2023 GCN
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset arxiv_2023 \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 2 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 0.3\
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# arxiv_2023 GCN + llm-based A-D
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN' dataset arxiv_2023 \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 2 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 0.3\
    gnn.train.sim_Add 0.6 gnn.train.sim_Delete 0.7

# arxiv_2023 GCN + llm-based LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset arxiv_2023 \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 2 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 0.3\
    gnn.train.sim_Add 1.1 gnn.train.sim_Delete -0.1

# arxiv_2023 GCN + llm-based A-D & LPA
CUDA_VISIBLE_DEVICES=0 python -m core.trainEnsemble gnn.train.feature_type TA_P_E gnn.model.name 'GCN_LPA' dataset arxiv_2023 \
    gnn.train.lr 0.01 gnn.model.hidden_dim 256 gnn.model.num_layers 3 gnn.train.dropout 0.5 gnn.train.num_lpa_iter 2 \
    gnn.train.lamda_lpa 1.0 gnn.train.lamda_lpa_2 0.3\
    gnn.train.sim_Add 0.3 gnn.train.sim_Delete 0.4