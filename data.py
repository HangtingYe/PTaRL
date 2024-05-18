import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import ipdb
import json
from joblib import Parallel, delayed
from transformers import BertTokenizer, BertTokenizerFast
import os
import math
import pandas as pd


def load(dataname, info, normalization, ratio):

    ## loading pretrain word2vectory
    if os.path.exists('./tokenizer'):
        tokenizer = BertTokenizerFast.from_pretrained('./tokenizer')
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained('./tokenizer')
    tokenizer.__dict__['model_max_length'] = 512

    task_type, n_num_features, n_cat_features, train_size, val_size, test_size = info.get('task_type'), info.get('n_num_features'), info.get('n_cat_features'), info.get('train_size'), info.get('val_size'), info.get('test_size')

    assert task_type in ['binclass', 'multiclass', 'regression']

    # preprocessing

    if ratio == 1.0:
        selected_rows = np.arange(train_size)
    else:
        selected_rows = np.random.choice(train_size, int(train_size * ratio), replace=False)
    train_size = int(train_size * ratio)

    ## numerical features
    N_train, N_val, N_test = np.load(f'./data/{dataname}/N_train.npy', allow_pickle=True), np.load(f'./data/{dataname}/N_val.npy', allow_pickle=True), np.load(f'./data/{dataname}/N_test.npy', allow_pickle=True)
    N_train = N_train[selected_rows, :]

    N = np.concatenate([N_train, N_val, N_test], axis=0).astype('float32')
    ### it has nan input for higgs_small
    N = np.nan_to_num(N)
    ### feature-wise normalize
    if normalization == 'standard':
        preprocess = sklearn.preprocessing.StandardScaler().fit(N[:train_size])
    elif normalization == 'minmax':
        preprocess = sklearn.preprocessing.MinMaxScaler().fit(N[:train_size])
    elif normalization == 'quantile':
        preprocess = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(N[:train_size])
    elif normalization == 'none':
        preprocess = sklearn.preprocessing.FunctionTransformer().fit(N[:train_size])
    ### N: (bs, cols)
    N = preprocess.transform(N)


    ## catergorical features
    if n_cat_features != 0:
        C_train, C_val, C_test = np.load(f'./data/{dataname}/C_train.npy', allow_pickle=True), np.load(f'./data/{dataname}/C_val.npy', allow_pickle=True), np.load(f'./data/{dataname}/C_test.npy', allow_pickle=True)
        C_train = C_train[selected_rows, :]

        C = np.concatenate([C_train, C_val, C_test], axis=0)

        C = [sklearn.preprocessing.LabelEncoder().fit_transform(C[:,i].reshape(-1,1)).astype('int64').reshape(-1,1) for i in range(C.shape[1])]
        C = np.concatenate(C, axis=1)

    else:
        C = None


    ## label
    y_train, y_val, y_test = np.load(f'./data/{dataname}/y_train.npy', allow_pickle=True), np.load(f'./data/{dataname}/y_val.npy', allow_pickle=True), np.load(f'./data/{dataname}/y_test.npy', allow_pickle=True)
    y_train = y_train[selected_rows]
    
    Y = np.concatenate([y_train, y_val, y_test], axis=0)
    ### regression
    if task_type == 'regression':
        Y = Y.astype('float32')
    ### classification
    else:
        Y = sklearn.preprocessing.LabelEncoder().fit_transform(Y).astype('int64')

    n_classes = int(max(Y)) + 1 if task_type == 'multiclass' else None
    ### !!! CRUCIAL for neural networks when solving regression problems !!!
    if task_type == 'regression':
        y_mean = Y[:train_size].mean().item()
        y_std = Y[:train_size].std().item()
        Y = (Y - y_mean) / y_std
    elif task_type == 'binclass':
        y_mean = y_std = None
        Y = Y.astype('float32')
    elif task_type == 'multiclass':
        y_mean = y_std = None
        Y = Y.astype('int64')

    # generate train, val, test
    X = {}
    y = {}

    if n_cat_features != 0:
        X_all = np.concatenate([N,C], axis=1)
        categories = np.max(C, axis=0) + 2
    else:
        X_all = N
        categories = None

    X['train'], X['val'], X['test'] = X_all[:train_size], X_all[train_size:train_size+val_size], X_all[-test_size:]
    y['train'], y['val'], y['test'] = Y[:train_size], Y[train_size:train_size+val_size], Y[-test_size:]

    X = {k: torch.tensor(v, dtype=torch.float).cuda() for k, v in X.items()}
    y = {k: torch.tensor(v).cuda() for k, v in y.items()}

    return X, y, n_classes, y_mean, y_std, categories