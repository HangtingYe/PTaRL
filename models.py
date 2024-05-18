from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import math
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch.nn.init as nn_init
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import json
from joblib import Parallel, delayed
import pandas as pd
from einops import rearrange, repeat
from sklearn.decomposition import PCA
import Models


class Model(nn.Module):
    def __init__(self, input_num, model_type, out_dim, info, topic_num, cluster_centers_, config, task_type, categories) -> None:
        super().__init__()

        self.input_num = input_num ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.info = info
        self.num_list = np.arange(info.get('n_num_features'))
        self.cat_list = np.arange(info.get('n_num_features'), info.get('n_num_features') + info.get('n_cat_features')) if info.get('n_cat_features')!=None else None
        self.topic_num = topic_num
        self.cluster_centers_ = cluster_centers_
        self.categories = categories

        self.config = config
        self.task_type = task_type

        self.build_model()



    def build_model(self):

        if self.model_type.split('_')[0] == 'MLP':
            # construct parameter for centers
            self.topic = nn.Parameter(torch.tensor(self.cluster_centers_), requires_grad=True)

            self.weight_ = nn.Parameter(torch.tensor(0.5))

            self.encoder = Models.mlp.MLP(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])

            self.head = nn.Linear(self.config['model']['d_layers'][-1], self.out_dim)

            self.reduce = nn.Sequential(
                        nn.Linear(self.config['model']['d_layers'][-1], self.config['model']['d_layers'][-1]),
                        nn.GELU(),
                        nn.Dropout(0.1), 
                        nn.Linear(self.config['model']['d_layers'][-1], self.config['model']['d_layers'][-1]),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.config['model']['d_layers'][-1], self.config['model']['d_layers'][-1]),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.config['model']['d_layers'][-1], self.topic_num)
                    )


    def forward(self, inputs_n, inputs_c):
        inputs_ = self.encoder(inputs_n, inputs_c)
        r_ = self.reduce(inputs_)
        if self.model_type.split('_')[1] == 'ot':
            return self.head(inputs_), torch.softmax(r_, dim=1), inputs_, torch.sigmoid(self.weight_)+0.01
        else:
            return self.head(inputs_)
        