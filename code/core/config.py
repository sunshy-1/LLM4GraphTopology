import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'cora'
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Number of runs with random init
    cfg.runs = 4
    cfg.gnn = CN()
    cfg.lm = CN()

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.name = 'GCN'
    # Number of gnn layers
    cfg.gnn.model.num_layers = 3
    # Hidden size of the model
    cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # Use PyG or DGL
    cfg.gnn.train.use_dgl = False
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0
    # Maximal number of epochs
    cfg.gnn.train.epochs = 200
    # Node feature type, options: ogb, TA, P, E
    cfg.gnn.train.feature_type = 'TA_P_E'
    # Number of epochs with no improvement after which training will be stopped
    cfg.gnn.train.early_stop = 50
    # Base learning rate
    cfg.gnn.train.lr = 0.01
    # L2 regularization, weight decay
    cfg.gnn.train.wd = 0.0
    # Dropout rate
    cfg.gnn.train.dropout = 0.0
    # Number of lpa iteration
    cfg.gnn.train.num_lpa_iter = 2
    # lamda lpa
    cfg.gnn.train.lamda_lpa = 1.0
    # lamda lpa2
    cfg.gnn.train.lamda_lpa_2 = 1.0
    # MIX
    cfg.gnn.train.MIX = False
    # use labels pred by gcn
    cfg.gnn.train.retrain_gcn = False
    # sim for add
    cfg.gnn.train.sim_Add = 0.0
    # sim for delete
    cfg.gnn.train.sim_Delete = 0.0

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    # LM model name
    cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.feat_shrink = ""

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    cfg.lm.train.batch_size = 9
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.epochs = 4
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 50000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4
    # Whether or not to use the gpt responses (i.e., explanation and prediction) as text input
    # If not, use the original text attributes (i.e., title and abstract)
    cfg.lm.train.use_gpt = False

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)
    
    global sim_Add 
    sim_Add = cfg.gnn.train.sim_Add
    global sim_Delete 
    sim_Delete = cfg.gnn.train.sim_Delete
    
    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
