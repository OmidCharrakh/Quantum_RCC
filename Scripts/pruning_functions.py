
import numpy as np
import pandas as pd
import networkx as nx
import torch as th
import copy
import torch.nn as nn

import distance_functions as dFun
import utilities_functions as uFun
import channel_functions as chFun
import rcc_functions as rcc
import simulator_functions as simFun
import random


def disconnected_adjacencies(A):
    row_ids, col_ids = np.nonzero(A)
    all_edges = [(i, j) for (i, j) in zip(row_ids, col_ids)]
    combinations = uFun.powerset(all_edges)
    adjacencies = [A]*len(combinations)
    for index, (adjacency, combination) in enumerate(zip(adjacencies, combinations)):
        adjacency = copy.deepcopy(adjacencies[index])
        for (i, j) in combination:
            adjacency[i, j] = 0
        adjacencies[index] = adjacency
    return adjacencies


def get_RandOp(A):
    n_nodes = A.shape[0]
    pairs = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
    labels = [uFun.label_reader(A, i, j, None, False) for (i, j) in pairs]
    RandOps = []
    for label in labels:
        if label == 0:
            operation = np.random.choice(['identity', 'add'], p=[8/10, 2/10])
        else:
            operation = np.random.choice(['identity', 'remove', 'reverse'], p=[3/10, 3/10, 4/10])
        RandOps.append(operation)
    return RandOps


def operate_DAG(A, operations):
    n_nodes = A.shape[0]
    pairs = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
    labels_old = [uFun.label_reader(A, i, j, None, False) for (i, j) in pairs]
    is_fine = False
    while is_fine is False:
        labels_new = []
        for (label_old, operation) in zip(labels_old, operations):
            if operation == 'identity':
                label_new = label_old
            if operation == 'add':
                label_new = np.random.choice([1, 2])
            elif operation == 'remove':
                label_new = 0
            elif operation == 'reverse':
                label_new = list(set([1, 2])-set([label_old]))[0]
            labels_new.append(label_new)
        A_op = uFun.list_Toadj(labels_new).astype(int)
        has_LowIncomming = A_op.sum(0).max() <= 5
        is_fine = has_LowIncomming
    return A_op


def generate_fakeDAGs(A_org, n_fakes):
    n1 = int(n_fakes/2)
    n2 = int((n_fakes-1)/2)
    fakes_1 = []
    i = 0
    while i < n1:
        A_fake = operate_DAG(A_org, get_RandOp(A_org))
        if nx.is_directed_acyclic_graph(nx.DiGraph(A_fake)):
            fakes_1.append(A_fake)
            i += 1
    fakes_2 = []
    As_disconnected = disconnected_adjacencies(A_org)
    for i in np.random.randint(0, len(As_disconnected), n2):
        A_fake = As_disconnected[i]
        fakes_2.append(A_fake)
    As_fake = [A_org] + fakes_1+fakes_2
    return th.tensor(np.array(As_fake))


def generate_fakeData(As_fake, n_points, simulator):
    n_fakes = len(As_fake)
    n_nodes = len(As_fake[0])
    datasets = th.zeros((n_fakes, n_points, 3*n_nodes))
    for index_fake in range(n_fakes):
        datasets[index_fake] = simulator(n_points, As_fake[index_fake])
    return datasets


def compute_fakeLoss(dataset_org, datasets_fake, n_runs=1):
    n_fakes = len(datasets_fake)
    n_points = dataset_org.shape[0]
    criterion = dFun.MMDloss(n_points)
    losses = th.zeros((n_fakes))
    for index_fake in range(n_fakes):
        loss = 0
        for run in range(n_runs):
            loss = loss + criterion(datasets_fake[index_fake], dataset_org).detach()
        losses[index_fake] = loss/n_runs
    return losses


def generate_fakeContainer(ch_container_org, n_fakesPerDAG, n_lossRuns, simulator, saving_path):
    container_d_org, container_a_org = ch_container_org
    container_a_org = container_a_org.numpy().astype(int)
    n_nodes = container_a_org[0].shape[0]
    n_samples, n_points = container_d_org.shape[0:2]
    container_loss = th.zeros((n_samples, n_fakesPerDAG))
    container_a_fake = th.zeros((n_samples, n_fakesPerDAG, n_nodes, n_nodes))
    for index_sample in range(n_samples):
        A_org = container_a_org[index_sample]
        As_fake = generate_fakeDAGs(A_org, n_fakesPerDAG)
        container_a_fake[index_sample] = As_fake
        dataset_org = container_d_org[index_sample]
        datasets_fake = generate_fakeData(As_fake, n_points, simulator)
        container_loss[index_sample] = compute_fakeLoss(dataset_org, datasets_fake, n_lossRuns)
        if np.mod(index_sample, 50) == 0:
            print(index_sample)
    container = (container_a_org, container_a_fake, container_loss)
    if saving_path is not None:
        th.save(container, saving_path)
    return container


def penaltyWg_evaluator(penalty_weight, container_fake):
    (container_a_org, container_a_fake, container_loss) = container_fake
    n_samples = len(container_a_org)
    deviations = []
    for index_sample in range(n_samples):
        A_org = container_a_org[index_sample]
        As_fake = container_a_fake[index_sample].numpy()
        losses = container_loss[index_sample].numpy()
        df = pd.DataFrame()
        df['A_fake'] = As_fake.tolist()
        df['n_edges'] = [int(A_fake.sum()) for A_fake in As_fake]
        df['n_wrongs'] = [int(np.abs(A_org-A_fake).sum()) for A_fake in As_fake]
        map_dict = {e: sorted(df['n_wrongs'].unique()).index(e) for e in sorted(df['n_wrongs'].unique())}
        df['rank_wrong'] = df['n_wrongs'].map(map_dict)
        df['loss'] = losses+penalty_weight*df['n_edges']
        # df = df.sort_values(by = ['loss'])
        n_rankWrongs = len(df['rank_wrong'].unique())
        df['rank_loss'] = pd.qcut(df['loss'], q=n_rankWrongs, labels=range(n_rankWrongs)).astype(int)
        # df['rank_loss'] = list(range(len(df)))
        # df = df.sort_index()
        df['deviation'] = (df['rank_wrong']-df['rank_loss']).abs()
        deviation = df['deviation'].mean()
        deviations.append(deviation)
    mean_deviation = np.mean(deviations)
    return df, mean_deviation


'''
def graph_evaluation(data_true, A_pred, penalty_weight=0, n_runs=10, shallow_simulator=None, simulator_path=None):
    n_points = data_true.shape[0]
    criterion = dFun.MMDloss(n_points)
    loss_list = []
    simulator = copy.deepcopy(shallow_simulator)
    simulator.reset_parameters()
    simulator.re_load(simulator_path)
    if not th.is_tensor(A_pred):
        A_pred = th.tensor(A_pred, dtype=th.int64)
    for run in range(n_runs):
        data_pred = simulator(n_points, A_pred).detach()
        loss = criterion(data_pred, data_true).item() + penalty_weight*A_pred.sum()
        loss_list.append(loss)
    return np.mean(loss_list)


def hill_climbing(data_true, A_pred, penalty_weight=0, n_runs=10, simulator=None, add=True, remove=True):
    if th.is_tensor(A_pred):
        G_pred = nx.DiGraph(A_pred.numpy())
    else:
        G_pred = nx.DiGraph(np.array(A_pred))
    nodelist = list(range(int(data_true.shape[1]/3)))
    test_Ms = [nx.adjacency_matrix(G_pred, nodelist=nodelist)]
    test_Ls = [graph_evaluation(data_true, test_Ms[0].todense(), penalty_weight, n_runs, simulator)]
    best_L = test_Ls[0]
    best_G = copy.deepcopy(G_pred)
    can_improve = True
    while can_improve:
        can_improve = False
        for (i, j) in best_G.edges():
            test_G = copy.deepcopy(best_G)
            if add:
                test_G.add_edge(j, i)
            if remove:
                test_G.remove_edge(i, j)
            test_M = nx.adjacency_matrix(test_G, nodelist=nodelist)
            if (nx.is_directed_acyclic_graph(test_G) and not any([(test_M != M).nnz == 0 for M in test_Ms])):
                test_L = graph_evaluation(data_true, test_M.todense(), penalty_weight, n_runs, simulator)
                test_Ms.append(test_M)
                test_Ls.append(test_L)
                if test_L < best_L:
                    can_improve = True
                    best_G = test_G
                    best_L = test_L
                    break
    best_A = np.array(nx.adjacency_matrix(best_G).todense())
    test_As = [np.array(test_Ms[index].todense()) for index in np.argsort(test_Ls)]
    test_Ls = np.sort(test_Ls)
    return (best_A, (test_As, test_Ls))


def hill_climbing(graph_evaluation, A_pred, data_true, simulator, penalty_weight=0, n_runs=1):
    G = nx.DiGraph(A_pred)
    best_score = graph_evaluation(G, data_true, simulator, penalty_weight, n_runs)
    best_G = copy.deepcopy(G)
    improved = True
    while improved:
        improved = False

        # Try reversing each edge in the graph
        for u, v in G.edges():
            G_rev = copy.deepcopy(G)
            G_rev.remove_edge(u, v)
            G_rev.add_edge(v, u)
            if nx.is_directed_acyclic_graph(G_rev):
                score = graph_evaluation(G_rev, data_true, simulator, penalty_weight, n_runs)
                if score > best_score:
                    best_score = score
                    best_G = copy.deepcopy(G_rev)
                    improved = True
        # Try removing each edge in the graph
        for u, v in G.edges():
            G_rm = copy.deepcopy(G)
            G_rm.remove_edge(u, v)
            if nx.is_directed_acyclic_graph(G_rm):
                score = graph_evaluation(G_rm, data_true, simulator, penalty_weight, n_runs)
                if score > best_score:
                    best_score = score
                    best_G = copy.deepcopy(G_rm)
                    improved = True

        # Try adding a random edge to the graph
        u = random.choice(list(G.nodes()))
        v = random.choice(list(G.nodes()))
        while G.has_edge(u, v):
            u = random.choice(list(G.nodes()))
            v = random.choice(list(G.nodes()))
        G_add = copy.deepcopy(G)
        G_add.add_edge(u, v)
        if nx.is_directed_acyclic_graph(G_add):
            score = graph_evaluation(G_add, data_true, simulator, penalty_weight, n_runs)
            if score > best_score:
                best_score = score
                best_G = copy.deepcopy(G_add)
                improved = True
        G = best_G
    A_hill = nx.to_numpy_array(best_G, dtype=int)
    return A_hill
'''


def hill_climbing(graph_evaluation, A_pred, data_true, simulator, penalty_weight=0, n_runs=1, max_iterations=10):
    G = nx.DiGraph(A_pred)
    best_G = copy.deepcopy(G)
    best_score = graph_evaluation(best_G, data_true, simulator, penalty_weight, n_runs)
    for i in range(max_iterations):
        # Select a random edge in the graph
        edges = list(best_G.edges())
        if not edges:
            break
        else:
            edge = random.choice(edges)
            # Try reversing the edge
            G_reverse = copy.deepcopy(best_G)
            G_reverse.remove_edge(*edge)
            G_reverse.add_edge(*edge[::-1])
            if nx.is_directed_acyclic_graph(G_reverse):
                reverse_score = graph_evaluation(G_reverse, data_true, simulator, penalty_weight, n_runs)
                if reverse_score > best_score:
                    best_G = G_reverse
                    best_score = reverse_score
            # Try removing the edge
            G_remove = copy.deepcopy(best_G)
            G_remove.remove_edge(*edge[::-1])
            if nx.is_directed_acyclic_graph(G_remove):
                remove_score = graph_evaluation(G_remove, data_true, simulator, penalty_weight, n_runs)
                if remove_score > best_score:
                    best_G = G_remove
                    best_score = remove_score
    A_hill = nx.to_numpy_array(best_G, dtype=int)
    return A_hill


def glb_to_loc(*labels_glb):
    labels_loc = []
    for A in labels_glb:
        n_nodes = A.shape[0]
        for (i, j) in [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]:
            if A[i, j] == 1:
                label = 1
            elif A[j, i] == 1:
                label = 2
            else:
                label = 0
            labels_loc.append(label)
    return labels_loc


def find_edges(A):
    edges = []
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                edges.append((i, j))
    return edges


def get_edge_importance(A, i, j, **kwargs):
    A_1 = copy.deepcopy(A)
    A_2 = copy.deepcopy(A)
    A_2[i, j] = 0
    s_1 = graph_evaluation(A_1, **kwargs)
    s_2 = graph_evaluation(A_2, **kwargs)
    s = (s_2 - s_1) / s_1
    return abs(s)


def get_edges_importance(A, **kwargs):
    importances = np.zeros((8, 8))
    edges = find_edges(A)
    for (i, j) in edges:
        importance = get_edge_importance(A=A, i=i, j=j, **kwargs)
        importances[i, j] = importance
    return importances


def graph_evaluation(pred, data_true, simulator, penalty_weight, n_runs):
    n_points = data_true.shape[0]
    criterion = dFun.MMDloss(n_points)
    if isinstance(pred, nx.DiGraph):
        A_pred = nx.to_numpy_array(pred, dtype=int)
    elif isinstance(pred, (np.ndarray, np.generic)):
        A_pred = pred
    if not th.is_tensor(A_pred):
        A_pred = th.tensor(A_pred, dtype=th.int64)
    loss_list = []
    for run in range(n_runs):
        data_pred = simulator(n_points, A_pred).detach()
        loss_1 = criterion(data_pred, data_true).item()
        loss_2 = penalty_weight*A_pred.sum()
        loss = loss_1 + loss_2
        loss_list.append(loss)
    return - np.mean(loss_list)
