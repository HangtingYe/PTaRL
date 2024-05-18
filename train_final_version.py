import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from data import load
from models import Model
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import argparse
import sklearn
import scipy
import gc
from sklearn.cluster import KMeans
import toml
import os
import copy

def build_data_loader(dataset, batch_size=128, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_loss_function(task_type):
    if task_type == 'regression': 
        loss_func = F.mse_loss
    elif task_type == 'binclass':
        loss_func = F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        loss_func = F.cross_entropy
    return loss_func


def run_one_epoch(model, data_loader, loss_func, model_type, config, ot_weight, diversity_weight, r_weight, optimizer=None):
   
    running_loss = 0.0

    for bid, (X_n, X_c, y) in enumerate(data_loader):
        
        if model_type.split('_')[1] == 'ot':
            pred, r_, hidden, weight_ = model(X_n, X_c)

            if loss_func == F.cross_entropy:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y.reshape(-1,1))
        
        else:
            pred = model(X_n, X_c)

            if loss_func == F.cross_entropy:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y.reshape(-1,1))

        if optimizer is not None and model_type.split('_')[1] == 'ot':

            norm = torch.mm(torch.sqrt(torch.sum(hidden**2, axis=1, keepdim=True)), torch.sqrt(torch.sum(model.topic.T**2, axis=0, keepdim=True)))

            loss_ot = torch.mean(torch.sum(r_*(torch.mm(hidden.float(), model.topic.T.float()) / norm), axis=1))
            loss += ot_weight * loss_ot

            selected_rows = np.random.choice(r_.shape[0], int(r_.shape[0] * 0.5), replace=False)

            distance = (r_[selected_rows].reshape(r_[selected_rows].shape[0],1,r_[selected_rows].shape[1])-r_[selected_rows]).abs().sum(dim=2)
            if loss_func == F.cross_entropy:
                label_similarity = (y.reshape(-1,1)[selected_rows] == y.reshape(-1,1)[selected_rows].T).float()
            else:
                y_min = min(y)
                y_max = max(y)
                num_bin = 1 + int(np.log2(y.shape[0]))
                interval_width = (y_max - y_min) / num_bin
                y_assign = torch.max(torch.tensor(0).cuda(),torch.min(((y.reshape(-1,1)-y_min)/interval_width).long(),torch.tensor(num_bin-1).cuda()))
                label_similarity = (y_assign.reshape(-1,1)[selected_rows] == y_assign.reshape(-1,1)[selected_rows].T).float()
            
            positive_mask = label_similarity
            positive_loss = torch.sum(distance * positive_mask) / (torch.sum(distance)+1e-8)
            loss_diversity = positive_loss

            loss += diversity_weight*loss_diversity

            # first should be sure that the the topic is learnable!
            r_1 = torch.sqrt(torch.sum(model.topic.float()**2,dim=1,keepdim=True))
            topic_metrix = torch.mm(model.topic.float(), model.topic.T.float()) / torch.mm(r_1, r_1.T)
            topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)

            l1 = torch.sum(topic_metrix.abs())
            l2 = torch.sum(topic_metrix ** 2)

            loss_sparse = l1 / l2
            loss_constraint = torch.abs(l1 - topic_metrix.shape[0])

            r_loss = loss_sparse + 0.5*loss_constraint
            
            loss += r_weight * r_loss

        else:
            None

        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss / len(data_loader)


def run_one_epoch_val(model, data_loader, loss_func, model_type, config, task_type, y_std):
    pred = []
    ground = []
    for bid, (X_n, X_c, y) in enumerate(data_loader):
        if model_type.split('_')[1] == 'ot':
            pred.append(model(X_n, X_c)[0].data.cpu().numpy())
        else:
            pred.append(model(X_n, X_c).data.cpu().numpy())
        ground.append(y)
    pred = np.concatenate(pred, axis=0)
    y = torch.cat(ground, dim=0)
    
    y = y.data.cpu().numpy()

    if task_type == 'binclass':
        pred = np.round(scipy.special.expit(pred))
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
        return -score
    elif task_type == 'multiclass':
        pred = pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
        return -score
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 * y_std
        return score



def fit(model, train_loader, val_loader, loss_func, model_type, config, task_type, y_std, ot_weight, diversity_weight, r_weight):
    best_val_loss = 1e30
    best_model = None

    # optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=config['training']['weight_decay'])

    early_stop = 20
    epochs = config['training']['n_epochs']

    patience = early_stop

    for eid in range(epochs):
        model.train()
        train_loss = run_one_epoch(
            model, train_loader, loss_func, model_type, config, ot_weight, diversity_weight, r_weight, optimizer
        )

        model.eval()
        val_loss = run_one_epoch_val(
            model, val_loader, loss_func, model_type, config, task_type, y_std
        )

        print(f'Epoch {eid}, train loss {train_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break
    return best_model

def test(model, test_loader, task_type, y_std, args, config):

    model.eval()

    pred = []
    ground = []
    for bid, (X_n, X_c, y) in enumerate(test_loader):
        if args.model_type.split('_')[1] == 'ot':
            pred.append(model(X_n, X_c)[0].data.cpu().numpy())
        else:
            pred.append(model(X_n, X_c).data.cpu().numpy())
        ground.append(y)
    pred = np.concatenate(pred, axis=0)
    y = torch.cat(ground, dim=0)
    
    y = y.data.cpu().numpy()

    if task_type == 'binclass':
        pred = np.round(scipy.special.expit(pred))
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'multiclass':
        pred = pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 * y_std

    print(f'test result, {score.item()}')

    np.save(open(f'./results_number/{args.dataname}_{args.model_type}_{args.ratio}_{args.hyper}_{args.seed}_{args.n_clusters}_{args.prototype_initial}.npy','wb'), score.item())
    torch.save(model.state_dict(), f'./models_number/{args.dataname}_{args.model_type}_{args.ratio}_{args.hyper}_{args.seed}_{args.n_clusters}_{args.prototype_initial}.pth')


def generate_topic(model, train_loader):
    hid_ = []
    for bid, (X_n, X_c, y) in enumerate(train_loader):
        hid = model.encoder(X_n, X_c)
        hid_.append(hid.data.cpu().numpy())
    hid_ = np.concatenate(hid_, axis=0)

    if args.prototype_initial == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_centers_ = kmeans.fit(hid_).cluster_centers_

    return cluster_centers_

    

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--hyper', type=str, default='default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', type=int, default=0)
    parser.add_argument('--ot_weight', type=float, default=0.25)
    parser.add_argument('--diversity_weight', type=float, default=0.25)
    parser.add_argument('--r_weight', type=float, default=0.25)
    parser.add_argument('--prototype_initial', type=str, default='kmeans')
    args = parser.parse_args()

    _set_seed(args.seed)

    config = toml.load(f'./hypers_{args.hyper}/{args.dataname}/{args.model_type}.toml')

    with open(f'./data/{args.dataname}/info.json') as f:
        info = json.load(f)

    gc.collect()
    torch.cuda.empty_cache()

    # X['train']: (bs, cols)
    X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'], args.ratio)
 
    task_type = info.get('task_type')
    print(task_type)

    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')

    train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cuda(), y['train']), config['training']['batch_size'], False)
    val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cuda(), y['val']), config['training']['batch_size'], False)
    test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cuda(), y['test']), config['training']['batch_size'], False)

    loss_func = get_loss_function(task_type)

    if args.n_clusters == 0:
        n_clusters = int(np.ceil(np.log2(n_num_features+n_cat_features)))
    else:
        n_clusters = args.n_clusters

    cluster_centers_ = np.zeros([n_clusters, 1])
    
    if args.model_type.split('_')[-1] == 'ot':
        source_model = args.model_type.split('_')[0]+'_'
        model = Model(n_num_features, source_model, n_classes if task_type == 'multiclass' else 1, info=info, 
        topic_num = n_clusters, cluster_centers_ = cluster_centers_, config = config, task_type = task_type, categories = categories)
        model.cuda()

        best_model = fit(model, train_loader, val_loader, loss_func, source_model, config, task_type, y_std, args.ot_weight, args.diversity_weight, args.r_weight)

        cluster_centers_ = generate_topic(best_model, train_loader)

        
    _set_seed(args.seed)
    model = Model(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, info=info,
    topic_num = n_clusters, cluster_centers_ = cluster_centers_, config = config, task_type = task_type, categories = categories)
    model.cuda()

    best_model = fit(model, train_loader, val_loader, loss_func, args.model_type, config, task_type, y_std, args.ot_weight, args.diversity_weight, args.r_weight)
    
    test(best_model, test_loader, task_type, y_std, args, config)




