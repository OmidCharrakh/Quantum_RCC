import numpy as np
import pandas as pd
import torch as th
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import ast
from pathlib import Path
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import sklearn
from sklearn.model_selection import train_test_split
import distance_functions as dFun
import utilities_functions as uFun

###########################################################
# Featurizer


class featurizer:

    def __init__(self, n_RandComps, scales, data_dim):

        self.n_RandComps = n_RandComps
        self.scales = scales
        self.data_dim = data_dim
        self.W = {n_vars: self.w_giver(n_vars) for n_vars in range(1, 10)}

    def w_giver(self, n_vars):
        n_RandComps = self.n_RandComps
        data_dim = self.data_dim
        scales = self.scales
        p1 = th.cat([scale*th.randn(n_RandComps, n_vars*data_dim) for scale in scales])
        p2 = 2 * np.pi * th.rand(n_RandComps*len(scales), 1)
        return th.cat((p1, p2), 1).T

    def kme(self, *xs):
        mu = []
        for x in xs:
            n_points, n_features = x.shape
            w_x = self.W[int(n_features/self.data_dim)]
            b_x = th.ones((n_points, 1))
            mu_x = th.mm(th.cat((x, b_x), 1), w_x).cos().mean(0)
            mu.append(mu_x)
        return th.cat(mu)


def get_all_triples(container_f, container_l):
    container_f, container_l = np.array(container_f, dtype=np.float32), np.array(container_l, dtype=int)
    n_samples = container_f.shape[0]
    n_pairs = container_f.shape[1]
    features = []
    labels = []
    for index_sample in range(n_samples):
        for index_pair in range(n_pairs):
            for index_triple in range(container_f.shape[2]):
                f = container_f[index_sample, index_pair, index_triple]
                features.append(f)
                l = container_l[index_sample, index_pair, index_triple]
                labels.append(l[0])
    X = np.array(features)
    y = np.array(labels)
    return (X, y)


def kme_featurizer(ch_container_path, featurizer_obj, scenario, quad_labeling=True, saving_path=None):
    container_d, container_a = th.load(ch_container_path)
    (n_samples, n_points, n_cols) = container_d.shape
    data_dim = featurizer_obj.data_dim
    n_RandComps = featurizer_obj.n_RandComps
    n_scales = len(featurizer_obj.scales)
    n_nodes = int(n_cols / data_dim)
    n_pairs = int((n_nodes ** 2 - n_nodes) / 2)
    n_triples = n_nodes-2
    dtype = th.float32
    if scenario in ['bi_ce', 'bi_ceccin']:
        container_f1 = th.zeros((n_samples, 3 * n_scales * n_RandComps), dtype=dtype)
        container_f2 = th.zeros((n_samples, 3 * n_scales * n_RandComps), dtype=dtype)
        container_l1 = th.zeros((n_samples, 1), dtype=dtype)
        container_l2 = th.zeros((n_samples, 1), dtype=dtype)
        cat_axis = 0
    else:
        container_f1 = th.zeros((n_samples, n_pairs, n_triples, 3 * n_scales * n_RandComps), dtype=dtype)
        container_f2 = th.zeros((n_samples, n_pairs, n_triples, 3 * n_scales * n_RandComps), dtype=dtype)
        container_l1 = th.zeros((n_samples, n_pairs, n_triples, 1), dtype=dtype)
        container_l2 = th.zeros((n_samples, n_pairs, n_triples, 1), dtype=dtype)
        cat_axis = 2
    for index_sample in range(n_samples):
        A = container_a[index_sample]
        ch_data = container_d[index_sample]
        if scenario in ['bi_ce', 'bi_ceccin']:
            x = ch_data[:, 0:3]
            y = ch_data[:, 3:6]
            container_f1[index_sample, :] = featurizer_obj.kme(x, y, th.cat((x, y), 1))
            container_f2[index_sample, :] = featurizer_obj.kme(y, x, th.cat((y, x), 1))
            container_l1[index_sample, :] = uFun.label_reader(A, 0, 1, None, quad_labeling)
            container_l2[index_sample, :] = uFun.label_reader(A, 1, 0, None, quad_labeling)
        else:
            for index_pair, (i, j) in enumerate([(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]):
                x = ch_data[:, data_dim * (i):data_dim * (i+1)]
                y = ch_data[:, data_dim * (j):data_dim * (j+1)]
                for index_triple, k in enumerate(list(set(range(n_nodes)) - set([i, j]))):
                    z = ch_data[:, data_dim * (k):data_dim * (k+1)]
                    container_f1[index_sample, index_pair, index_triple, :] = featurizer_obj.kme(x, y, th.cat((x, y, z), 1))
                    container_f2[index_sample, index_pair, index_triple, :] = featurizer_obj.kme(y, x, th.cat((y, x, z), 1))
                    container_l1[index_sample, index_pair, index_triple, :] = uFun.label_reader(A, i, j, k, quad_labeling)
                    container_l2[index_sample, index_pair, index_triple, :] = uFun.label_reader(A, j, i, k, quad_labeling)
        if np.mod(index_sample, 100) == 0:
            print(index_sample)
    container_f = th.cat((container_f1, container_f2), cat_axis)
    container_l = th.cat((container_l1, container_l2), cat_axis)
    kme_container = (container_f, container_l)
    if saving_path is not None:
        th.save(kme_container, saving_path)
    return kme_container


def container_to_df(kme_container_path, scenario, i0=None, i1=None, return_unique=False):
    container_f, container_l = th.load(kme_container_path)
    if return_unique:
        unique_ids = np.unique(container_l, axis=0, return_index=True)[1]
        container_f = container_f[unique_ids]
        container_l = container_l[unique_ids]
    if (i0 is not None) and (i1 is not None):
        container_f, container_l = container_f[i0:i1], container_l[i0:i1]
    if scenario in ['bi_ce', 'bi_ceccin']:
        df = pd.DataFrame(th.cat([container_l, container_l, container_f], 1))
    elif scenario in ['multi_8node', 'multi_base']:
        X, y = get_all_triples(container_f, container_l)
        df = pd.DataFrame(np.concatenate((y.reshape(-1, 1), y.reshape(-1, 1), X), axis=1))
    df.rename(columns={0: 'label_1', 1: 'label_2'}, inplace=True)
    df['label_2'] = uFun.convert_labels(df['label_1'], uFun.get_MapDict(1))
    return df


def get_best_treshold(clf, kme_container_path, n_DAGs, n_tresholds=100, saving_path=None):
    container_f, container_l = th.load(kme_container_path)
    n_samples, n_pairs = container_f.shape[0], container_f.shape[1]
    n_triples = int(container_f.shape[2]/2)
    n_nodes = n_triples+2
    node_pairs = dict(zip(list(range(n_pairs)), [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]))
    thresholds = np.linspace(0.5, 1, n_tresholds)
    if (n_DAGs is None) or (n_DAGs > n_samples):
        n_DAGs = n_samples
    w_array = np.zeros((n_DAGs, n_tresholds))
    for index_sample in range(n_DAGs):
        P0, P1, P2, A_pred = np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes))
        for index_pair, (i, j) in node_pairs.items():
            P0[i, j], P1[i, j], P2[i, j] = uFun.apply_clf(x=container_f[index_sample, index_pair, 0:n_triples], clf=clf, normalize=True)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                probs = np.array([P0[i, j], P1[i, j], P2[i, j]])
                val_max = np.max(probs)
                pos_max = np.argmax(probs)
                if pos_max == 1:
                    A_pred[i, j] = val_max
                elif pos_max == 2:
                    A_pred[j, i] = val_max
        A_true = uFun.list_Toadj(container_l[index_sample, :, 0, 0])
        w_array[index_sample, :] = [int(np.abs(A_true-np.where(A_pred > threshold, 1, 0)).sum()) for threshold in thresholds]
    wrong_preds = w_array.sum(0)/n_DAGs
    df = pd.DataFrame()
    df['wrong_preds'] = wrong_preds
    df['treshold'] = thresholds
    df.to_csv(saving_path, index=False)
    print('Best performance happens at threshold {:.04f} with average {:.04f} wrong prediction'.format(df.loc[df['wrong_preds'].argmin(), 'treshold'], df['wrong_preds'].min()))
    plt.plot(df['treshold'], df['wrong_preds'])
    plt.show()
    plt.close()
    return df
