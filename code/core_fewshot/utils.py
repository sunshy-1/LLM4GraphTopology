import os
import numpy as np
import time
import datetime
import pytz
import torch


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return

    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper

def one_hot_embedding(labels, num_classes):
    result = []
    y = torch.eye(num_classes)

    for label in labels:
        if label == -99:
            result.append(torch.zeros(y.shape[1]))
        else:
            result.append(y[label])
    
    return torch.stack(result)

def get_few_shot_samples(data, node_id, numTrain_perclass = 20, numVal=500, numTest=1000):
    num_classs = max(data.y) + 1
    class_num_cnt = torch.ones(num_classs) * numTrain_perclass
    
    train_node, val_node, test_node  = [], [], []

    for node_id in node_id:
        label_node = data.y[node_id]
        if class_num_cnt[label_node] > 0:
            train_node.append(node_id)
            class_num_cnt[label_node] -= 1
        elif len(val_node) < numVal:
            val_node.append(node_id)
        elif len(test_node) < numTest:
            test_node.append(node_id)
        else:
            break
    train_node, val_node, test_node = np.array(train_node), np.array(val_node), np.array(test_node)

    return train_node, val_node, test_node