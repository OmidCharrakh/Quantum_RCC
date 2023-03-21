import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
import sklearn
import itertools
import torch as th
from torch.utils.data import DataLoader
import networkx as nx
import IPython
import ast
import scipy
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import (PCA, KernelPCA, FastICA)
import sklearn.decomposition
sns.set()


def object_saver(obj, path):
    if path is None:
        pass
    else:
        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def object_loader(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def literal_return(val):
    try:
        return ast.literal_eval(val)
    except ValueError:
        return (val)


class EarlyStopping:
    def __init__(
            self, patience, checkpoint_path, minimize_score=True, **kwargs
            ):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.minimize_score = minimize_score
        self.checkpoint_path = checkpoint_path
        try:
            os.remove(self.checkpoint_path)
        except:
            pass
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, val_score, model):
        score = -val_score if self.minimize_score else +val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.best_score > score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        try:
            th.save(model.state_dict(), self.checkpoint_path)
        except:
            th.save(model, self.checkpoint_path)


class Live_Visualizer:
    def __init__(self, mode, metrics=[['accuracy', 'recall'], ['recall_0', 'recall_1', 'recall_2'], ['f1_score_0', 'f1_score_1', 'f1_score_2']], saving_path=None):
        self.mode = mode
        self.metrics = metrics
        self.saving_path = saving_path
        for axis_metrics in self.metrics:
            for metric in axis_metrics:
                setattr(self, metric, [])

    def visualize(self, logs, saving_path):
        for axis_metrics in self.metrics:
            for metric in axis_metrics:
                getattr(self, metric).append(logs.get(metric))
        if self.mode == 'on':
            n_axes = len(self.metrics)
            IPython.display.clear_output(wait=True)
            fig, axes = plt.subplots(nrows=1, ncols=n_axes, figsize=(5*n_axes, 5), tight_layout=True)
            for index_axis, axis_metrics in enumerate(self.metrics):
                for metric in axis_metrics:
                    metric_value = getattr(self, metric)
                    index_run = len(metric_value)
                    axes[index_axis].plot(metric_value, label=metric)
                    axes[index_axis].scatter(range(index_run), metric_value)
                axes[index_axis].legend()
            if saving_path is not None:
                plt.savefig(saving_path, dpi=300)
            plt.show()
            plt.close()


class MLPClassifier_torch(th.nn.Module):
    def __init__(self, hidden_layer_sizes=(300, 100),
                 batch_size=1000,
                 n_epochs=100,
                 patience=50,
                 lr=.001,
                 checkpoint_path='checkpoint.pt',
                 visualize_progress=True,
                 criterion_train=th.nn.CrossEntropyLoss(),
                 criterion_valid=th.nn.CrossEntropyLoss()
                 ):
        super(MLPClassifier_torch, self).__init__()
        if isinstance(hidden_layer_sizes, tuple):
            self.hidden_layer_sizes = list(hidden_layer_sizes)
        else:
            self.hidden_layer_sizes = [hidden_layer_sizes]
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        self.visualize_progress = visualize_progress
        self.criterion_train = criterion_train
        self.criterion_valid = criterion_valid
        for index_nh, nh in enumerate(self.hidden_layer_sizes):
            setattr(self, 'nh_{}'.format(index_nh), nh)

    def forward(self, x):
        for index_nh in range(0, 1 + len(self.hidden_layer_sizes)):
            layer = getattr(self, 'layer_{}'.format(index_nh))
            if index_nh < len(self.hidden_layer_sizes):
                x = th.nn.ReLU()(layer(x))
            else:
                x = layer(x)
        return x

    def predict(self, x):
        if not th.is_tensor(x):
            x = th.tensor(x, dtype=th.float32)
        x = self.forward(x)
        y_pred = th.nn.Softmax(1)(x).argmax(1)
        return y_pred

    def predict_proba(self, x):
        x = th.tensor(x, dtype=th.float32)
        x = self.forward(x)
        y_prob = th.nn.Softmax(1)(x)
        return y_prob

    def fit(self, data_train, data_valid):
        X_train, y_train = data_train
        X_valid, y_valid = data_valid
        n_feature = X_train.shape[1]
        n_class = len(np.unique(y_train))
        for index_nh in range(0, 1 + len(self.hidden_layer_sizes)):
            if index_nh == 0:
                layer = th.nn.Linear(n_feature, getattr(self, 'nh_{}'.format(index_nh)))
                th.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif index_nh < len(self.hidden_layer_sizes):
                layer = th.nn.Linear(getattr(self, 'nh_{}'.format(index_nh-1)), getattr(self, 'nh_{}'.format(index_nh)))
                th.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                layer = th.nn.Linear(self.hidden_layer_sizes[-1], n_class)
                th.nn.init.xavier_uniform_(layer.weight)
            setattr(self, 'layer_{}'.format(index_nh), layer)

        data_train = np.concatenate((y_train.reshape(-1, 1), X_train), 1)
        data_train = th.tensor(data_train, dtype=th.float32)
        data_valid = np.concatenate((y_valid.reshape(-1, 1), X_valid), 1)
        data_valid = th.tensor(data_valid, dtype=th.float32)
        dataloader_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        dataloader_valid = DataLoader(data_valid, batch_size=self.batch_size, shuffle=True)
        early_stopping = EarlyStopping(patience=self.patience, checkpoint_path=self.checkpoint_path, minimize_score=True)
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        prog_train = []
        prog_valid = []
        for epoch in range(self.n_epochs):
            loss_train = []
            loss_valid = []
            for data in dataloader_train:
                X, y = data[:, 1:], data[:, 0].type(th.LongTensor)
                optimizer.zero_grad()
                yhat = self.forward(X)
                loss = self.criterion_train(yhat, y)
                loss.backward()
                optimizer.step()
                loss_train.append(loss.item())
            prog_train.append(np.mean(loss_train))
            with th.no_grad():
                self.eval()
                for data in dataloader_valid:
                    X, y = data[:, 1:], data[:, 0].type(th.LongTensor)
                    yhat = self.forward(X)
                    loss = self.criterion_valid(yhat, y)
                    loss_valid.append(loss.item())
                prog_valid.append(np.mean(loss_valid))
                early_stopping(np.mean(loss_valid), self)
                if early_stopping.early_stop:
                    print('EarlyStopping')
                    break
            if self.visualize_progress:
                IPython.display.clear_output(wait=True)
                with plt.style.context('seaborn-white'):
                    plt.figure(figsize=(16, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(range(1 + epoch), prog_train)
                    plt.scatter(range(1 + epoch), prog_train)
                    plt.title('train')
                    plt.xlabel('number of epochs')
                    plt.ylabel('')
                    plt.subplot(1, 3, 2)
                    plt.plot(range(1 + epoch), prog_valid)
                    plt.scatter(range(1 + epoch), prog_valid)
                    plt.title('valid')
                    plt.xlabel('number of epochs')
                    plt.ylabel('')
                    IPython.display.display(plt.gcf())
                    plt.close('all')
        self.load_state_dict(th.load(self.checkpoint_path))
        return self


def adjacencies_splitter(adjacencies_path, splitting_ratios, saving_path):
    train_ratio, valid_ratio, test_ratio = splitting_ratios
    n_nodes = int(np.sqrt(np.loadtxt(adjacencies_path).shape[1]))
    adjacencies = [A.reshape(n_nodes, n_nodes) for A in np.loadtxt(adjacencies_path)]
    n_edges = [A.sum() for A in adjacencies]
    index_train, index_test, n_edges_train, n_edges_test = train_test_split(list(range(len(adjacencies))), n_edges, test_size=1 - train_ratio, stratify=n_edges, random_state=0)
    index_valid, index_test, n_edges_valid, n_edges_test = train_test_split(index_test, n_edges_test, test_size=test_ratio/(test_ratio + valid_ratio), stratify=n_edges_test, random_state=0)
    adjacencies_train = [adjacencies[index] for index in index_train]
    adjacencies_valid = [adjacencies[index] for index in index_valid]
    adjacencies_test = [adjacencies[index] for index in index_test]
    splitted_adjacencies = (adjacencies_train, adjacencies_valid, adjacencies_test)
    object_saver(splitted_adjacencies, saving_path)
    return splitted_adjacencies


def graph_plotter(*As, starting_index=0, saving_path=None):
    sns.set_style("dark")
    n_graphs = len(As)
    fig, axes = plt.subplots(nrows=1, ncols=n_graphs, figsize=(4*n_graphs, 3))
    for index_A, A in enumerate(As):
        G = nx.MultiDiGraph(A)
        G = nx.relabel_nodes(G, {index: r'$\rho_{}$'.format(index + starting_index) for index in range(len(A))})
        ax = axes if n_graphs == 1 else axes[index_A]
        nx.draw_networkx(G, ax=ax, pos=nx.circular_layout(G), node_color='gold', node_size=1000, width=1.5, font_size=15)
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300)
    plt.show()
    plt.close()


def label_reader(A, i, j, k, quad_labeling):
    A = np.array(A)
    if A[i, j] == 1:
        label = 1
    elif A[j, i] == 1:
        label = 2
    elif quad_labeling and (k is None) and (not nx.d_separated(nx.DiGraph(A), {i}, {j}, {})):
        label = 3
    elif quad_labeling and (k is not None) and (not nx.d_separated(nx.DiGraph(A), {i}, {j}, {k})):
        label = 3
    else:
        label = 0
    return label


def adj_Tolist(A, quad_labeling, return_th_list=False):
    n_nodes = A.shape[0]
    labels = []
    for (i, j) in [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]:
        for k in list(set(range(n_nodes)) - set([i, j])):
            label = label_reader(A, i, j, k, quad_labeling)
            labels.append(label)
    if return_th_list:
        labels = [th.tensor(label, dtype=th.int64).reshape(1) for label in labels]
    return labels


def list_Toadj(lst):
    n_pairs = len(lst)
    n_nodes = {int((n**2-n)/2): n for n in range(20)}.get(n_pairs)
    node_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    A = np.zeros((n_nodes, n_nodes))
    for index_pair, pair in enumerate(node_pairs):
        (i, j) = pair
        if lst[index_pair] == 1:
            A[i, j] = 1
        elif lst[index_pair] == 2:
            A[j, i] = 1
        else:
            A[i, j] = 0
    return A


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    lists = [[ss for mask, ss in zip(masks, s) if i & mask] for i in range(1 << x)]
    lists.sort(key=len)
    return lists


def weighted_sampler(df, n_samples, class_weights=None, target_column=1):
    n_classes = len(np.unique(df.iloc[:, target_column]))
    if n_samples == -1:
        df_sample = df
    elif class_weights is None:
        df_sample = df.sample(n_samples)
    else:
        class_weights = np.array(class_weights)/np.sum(class_weights)
        samples_perClass = [int(n_samples*weight) for weight in class_weights]
        df_sample = pd.DataFrame()
        for index_class, n_samples in enumerate(samples_perClass):
            df_masked = df[df.iloc[:, target_column] == index_class].sample(n_samples)
            df_sample = pd.concat([df_sample, df_masked], axis=0)
    indecies = sorted(df_sample.index.tolist())
    df_sample = df_sample.reset_index(drop=True)
    print([(df_sample.iloc[:, target_column] == index_class).sum() for index_class in range(n_classes)])
    return df_sample, indecies


def clf_evaluator_meta(kme_container, clf, n_DAGs=-1, percentile=80, predict_both_directions=True, saving_path=None):
    container_f, container_l = kme_container
    n_samples, n_pairs, n_triples, n_features = container_f.shape
    container_l = container_l[:, :, :int(n_triples/2), :].to(th.long)
    n_nodes = {int((n**2-n)/2): n for n in range(20)}.get(n_pairs)
    if (n_DAGs > n_samples) or (n_DAGs == -1):
        n_DAGs = n_samples
    node_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    clf_classes = sorted(clf.classes_)
    n_classes = len(clf_classes)
    if n_classes == 3:
        container_l = convert_labels(container_l, {0: 0, 1: 1, 2: 2, 3: 0})
        quad_labeling = False
    elif n_classes == 4:
        quad_labeling = True
    df = pd.DataFrame()
    df['nWrongs_raw'] = None
    df['nWrongs_percentile'] = None
    df['nEdges_true'] = None
    df['nEdges_pred'] = None
    df['confusion'] = None
    df['A_true'] = None
    df['A_pred'] = None
    df['A_binarized'] = None
    df['A_percentile'] = None
    for index_sample in range(n_DAGs):
        A_pred = np.zeros((n_nodes, n_nodes))
        for index_pair, (i, j) in enumerate(node_pairs):
            if predict_both_directions:
                x1 = container_f[index_sample, index_pair, :int(n_triples/2), :]
                x2 = container_f[index_sample, index_pair, int(n_triples/2):, :]
                p1 = apply_clf(x=x1, clf=clf, normalize=False)
                p2 = apply_clf(x=x2, clf=clf, normalize=False)
                p2 = np.array([p2[0], p2[2], p2[1]])
                probs = p1 + p2
                probs /= probs.sum()
            else:
                x1 = container_f[index_sample, index_pair, :int(n_triples/2), :]
                probs = apply_clf(x=x1, clf=clf, normalize=True,)
            if probs.argmax() == 1:
                A_pred[i, j] = probs.max()
            elif probs.argmax() == 2:
                A_pred[j, i] = probs.max()

        A_true = list_Toadj(container_l[index_sample, :, 0])
        A_binarized = np.where(A_pred > 0, 1, 0)
        A_percentile = np.where((A_pred > np.percentile(A_pred, percentile)) * A_pred > 0, 1, 0)
        nWrongs_raw = int(np.abs(A_true - A_binarized).sum())
        nWrongs_percentile = int(np.abs(A_true - A_percentile).sum())
        vector_true = adj_Tolist(A_true, quad_labeling, False)
        vector_pred = adj_Tolist(A_binarized, quad_labeling, False)
        confusion = sklearn.metrics.confusion_matrix(vector_true, vector_pred, labels=clf_classes, normalize='true')
        df.loc[index_sample, 'nWrongs_raw'] = nWrongs_raw
        df.loc[index_sample, 'nWrongs_percentile'] = nWrongs_percentile
        df.loc[index_sample, 'nEdges_true'] = int(A_true.sum())
        df.loc[index_sample, 'nEdges_pred'] = int(A_binarized.sum())
        df.at[index_sample, 'confusion'] = confusion
        df.at[index_sample, 'A_true'] = A_true
        df.at[index_sample, 'A_pred'] = A_pred
        df.at[index_sample, 'A_binarized'] = A_binarized
        df.at[index_sample, 'A_percentile'] = A_percentile
        if np.mod(index_sample, 50) == 0:
            print(index_sample)
    IPython.display.clear_output(wait=True)
    print('Average wrong predictions on {} DAGs => \nRaw (no post-processing): {:.4}\nPercentile: {:.4}'.format(n_DAGs, df['nWrongs_raw'].mean(), df['nWrongs_percentile'].mean()))
    if saving_path is not None:
        object_saver(df, saving_path)
    return df


def apply_clf(x, clf, normalize=True):
    x = x.numpy()
    x = clf.predict_proba(x)
    if normalize:
        x = x.mean(0)
    else:
        x = x.sum(0)
    return x


def clf_report(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)
    confusion_n = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')
    return (accuracy, confusion, confusion_n)


def reg_report(y_true, y_pred):
    EVS = sklearn.metrics.explained_variance_score(y_true, y_pred)
    MAE = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    MSE = sklearn.metrics.mean_squared_error(y_true, y_pred)
    return (EVS, MAE, MSE)


def clf_performance(clf, X_test, y_test):
    X_true = X_test
    y_true = y_test
    y_pred = clf.predict(X_true)
    n_class = len(np.unique(y_true))
    dict_0 = {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
        'f1_score': sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': sklearn.metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': sklearn.metrics.recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
    dict_1 = {
        'recall_{}'.format(c): sklearn.metrics.recall_score(y_true, y_pred, average='macro', labels=[c], zero_division=0) for c in range(n_class)
        }
    dict_2 = {
        'precision_{}'.format(c): sklearn.metrics.precision_score(y_true, y_pred, average='macro', labels=[c], zero_division=0) for c in range(n_class)
        }
    dict_3 = {
        'f1_score_{}'.format(c): sklearn.metrics.f1_score(y_true, y_pred, average='macro', labels=[c], zero_division=0) for c in range(n_class)
        }
    dict_final = {
        metric_name: metric_value for dict_ in [dict_0, dict_1, dict_2, dict_3] for metric_name, metric_value in dict_.items()
        }
    return dict_final


def clf_tuner(X_train, y_train,
              base_estimator=MLPClassifier(hidden_layer_sizes=(100, 50), solver='adam', activation='tanh', validation_fraction=0.1, early_stopping=True, n_iter_no_change=10, max_iter=1000),
              search_space=dict(alpha=scipy.stats.uniform(1e-5, 1e-3), batch_size=scipy.stats.randint(50, 1000), learning_rate_init=scipy.stats.uniform(1e-5, 1e-3)),
              objective='balanced_accuracy',
              n_combinations=100,
              n_cvs=3,
              saving_path=None,
              ):
    tune_object = sklearn.model_selection.RandomizedSearchCV(
        base_estimator,
        search_space,
        scoring=objective,
        refit=objective,
        n_iter=n_combinations,
        cv=n_cvs,
        n_jobs=-1,
        verbose=1,
        )
    analysis = tune_object.fit(X_train, y_train)
    print(analysis.best_params_)
    object_saver(analysis, saving_path)
    return analysis


def pca_visualizer(X, y,
                   transformers=[
                       PCA(n_components=2),
                       KernelPCA(n_components=2, kernel='rbf', gamma=1),
                       FastICA(n_components=2)]):
    n_transformers = len(transformers)
    labels = np.unique(y)
    colors = ['r', 'g', 'b', 'y']
    fig, axes = plt.subplots(nrows=1, ncols=n_transformers, figsize=(4 * n_transformers, 5), tight_layout=True)
    for index_transformer, transformer in enumerate(transformers):
        principalComponents = transformer.fit_transform(X)
        df = pd.DataFrame(principalComponents, columns=['component_1', 'component_2'])
        df['label'] = y
        for index_label, label in enumerate(labels):
            axes[index_transformer].scatter(
                df.loc[df['label'] == label, 'component_1'],
                df.loc[df['label'] == label, 'component_2'],
                c=colors[index_label], s=15)
        axes[index_transformer].legend(labels)
    plt.show()
    plt.close()


def convert_labels(x, map_dict):
    if isinstance(x, list):
        x = np.array(x)
        x = np.vectorize(map_dict.get)(x)
        x = list(x)
    if isinstance(x, np.ndarray):
        x = np.vectorize(map_dict.get)(x)
    if isinstance(x, pd.Series):
        x = x.map(map_dict)
    if th.is_tensor(x):
        x = np.array(x)
        x = np.vectorize(map_dict.get)(x)
        x = th.tensor(x)
    return x


def get_MapDict(strategy):
    if strategy == 0:
        map_dict = {0: 0, 1: 1, 2: 2, 3: 3}
    elif strategy == 1:
        map_dict = {0: 0, 1: 1, 2: 2, 3: 0}
    elif strategy == 2:
        map_dict = {0: 0, 1: 1, 2: 1, 3: 0}
    elif strategy == 3:
        map_dict = {0: 1, 1: 0, 2: 0, 3: 0}
    elif strategy == 4:
        map_dict = {0: 0, 1: 0, 2: 0, 3: 1}
    return map_dict


def confusion_plotter(y_test, y_pred, saving_path=None):
    label_annotation_dict = {0: '$x\perp\!\!\!\perp y$', 1: '$x\\rightarrow y$', 2: '$x\leftarrow y$', 3: '$x\leftarrow z\\rightarrow y$'}
    plot_labels = [label_annotation_dict[label] for label in np.unique(y_test)]
    real_labels = [label for label in np.unique(y_test)]
    n_classes = len(real_labels)
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=real_labels, normalize='true')
    print('Accuracy: {:.03f}'.format(sklearn.metrics.accuracy_score(y_test, y_pred)))
    sns.set_theme(style='white')
    plt.figure(figsize=(5, 5))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(np.arange(n_classes), plot_labels, fontsize=14)
    plt.yticks(np.arange(n_classes), plot_labels, fontsize=14)
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], '.2f'), fontsize=14, horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max()/2. else "black")
        plt.tight_layout()
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300)
    plt.show()
    plt.close()


def predict_graph(features, clf, percentile=80, predict_both_directions=True):
    n_pairs, n_triples, n_features = features.shape
    n_nodes = {int((n**2 - n)/2): n for n in range(20)}.get(n_pairs)
    A_pred = np.zeros((n_nodes, n_nodes))
    node_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    for index_pair, (i, j) in enumerate(node_pairs):
        if predict_both_directions:
            x1 = features[index_pair, :int(n_triples/2), :]
            x2 = features[index_pair, int(n_triples/2):, :]
            p1 = apply_clf(x=x1, clf=clf, normalize=False)
            p2 = apply_clf(x=x2, clf=clf, normalize=False)
            p2 = np.array([p2[0], p2[2], p2[1]])
            probs = p1 + p2
            probs /= probs.sum()
        else:
            x = features[index_pair, :int(n_triples/2), :]
            probs = apply_clf(x=x, clf=clf, normalize=True,)
        if probs.argmax() == 1:
            A_pred[i, j] = probs.max()
        elif probs.argmax() == 2:
            A_pred[j, i] = probs.max()
    A_percentile = np.where((A_pred > np.percentile(A_pred, percentile)) * A_pred > 0, 1, 0)
    return A_percentile


def save_as_pickle(file, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def split_1cc_dags(adjancencies_path, saving_paths):
    '''
    Find all one_cc DAGS =>
    Split them stratified on number of edges =>
    Save them
    '''

    candidates = np.loadtxt(adjancencies_path).astype(int)
    unique_ids = np.unique(candidates, axis=0, return_index=True)[1]
    candidates = candidates[unique_ids]
    adjacencies = []
    for candidate in candidates:
        A = candidate.reshape(8, 8)
        if max_ccs(A) == 1:
            adjacencies.append(A)
    edges = [A.sum() for A in adjacencies]

    adjacencies_tr, adjacencies_xx, edges_tr, edges_xx = train_test_split(
        adjacencies,
        edges,
        train_size=0.60,
        stratify=edges,
        random_state=0,
        )
    adjacencies_va, adjacencies_te, edges_va, edges_te = train_test_split(
        adjacencies_xx,
        edges_xx,
        test_size=0.5,
        stratify=edges_xx,
        random_state=0,
        )
    As_list = [adjacencies_tr, adjacencies_va, adjacencies_te]
    for index_set, adjacencies in enumerate(As_list):
        contrainer = np.zeros((len(adjacencies), 64))
        for index_adj, adj in enumerate(adjacencies):
            contrainer[index_adj] = adj.reshape(-1)
        np.savetxt(saving_paths[index_set], contrainer, fmt='%s')
    return (adjacencies_tr, adjacencies_va, adjacencies_te)


def max_ccs(A):
    'Finds the max number of CC for all pair nodes in an adjacency matrix'
    n_nodes = A.shape[0]
    n_ccs = []
    for i in range(n_nodes):
        for j in range(1 + i, n_nodes):
            n = np.sum([A[k, i]*A[k, j] for k in range(n_nodes)])
            n_ccs.append(n)
    return max(n_ccs)


def embed_graph(i, j, k, A_base, n_nodes):
    A = th.zeros((n_nodes, n_nodes), dtype=th.int32)
    node_names = {0: i, 1: j, 2: k}
    for (c0, c1) in th.nonzero(A_base):
        n0 = node_names[c0.item()]
        n1 = node_names[c1.item()]
        A[n0, n1] = 1
    return A


def embed_graph2(adjacency_base, node_ids_base, n_nodes_embedding):
    A = th.zeros((n_nodes_embedding, n_nodes_embedding), dtype=th.int32)
    node_names = {n: i for n, i in enumerate(node_ids_base)}
    for (c0, c1) in th.nonzero(adjacency_base):
        n0 = node_names[c0.item()]
        n1 = node_names[c1.item()]
        A[n0, n1] = 1
    return A


def get_list_ijk(n_nodes):
    adjacencies_base = [
        th.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        th.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        th.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        th.tensor([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
        th.tensor([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
        th.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        th.tensor([[0, 1, 1], [0, 0, 0], [0, 1, 0]]),
        th.tensor([[0, 0, 1], [1, 0, 1], [0, 0, 0]]),
    ]
    list_ijk = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in range(n_nodes):
                if (i - j)*(j - k)*(k - i):
                    for A_base in adjacencies_base:
                        list_ijk.append((i, j, k))
    return list_ijk


def remove_edge(adj_matrix):
    n = len(adj_matrix)
    adj_matrices = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                new_adj_matrix = adj_matrix.copy()
                new_adj_matrix[i][j] = 0
                adj_matrices.append(new_adj_matrix)
    return adj_matrices


def plot_confusion(ax, confusion, annotations, **kwargs):

    anotations_labels_dict = {
        0: '$\mathbf{X}\\rightarrow\mathbf{Y}$',
        1: '$\mathbf{X}\leftarrow\mathbf{Y}$',
        2: '$\mathbf{X}\perp\!\!\!\!\!\perp\mathbf{Y}$',
        3: '$\mathbf{X}\leftarrow\mathbf{Z}\\rightarrow \mathbf{Y}$',
    }

    labels = [anotations_labels_dict[anotation] for anotation in annotations]
    n_labels = len(labels)
    if 'fontsize' in kwargs.keys():
        fontsize = kwargs['fontsize']
    else:
        fontsize = 12
    ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(n_labels), labels, fontsize=fontsize,)
    ax.set_yticks(np.arange(n_labels), labels, fontsize=10)
    for i, j in itertools.product(range(n_labels), range(n_labels)):
        ax.text(
            j, i, format(confusion[i, j], '.2f'),
            fontsize=fontsize,
            horizontalalignment="center",
            fontweight="bold" if confusion[i, j] > confusion.max()/2. else None,
            color="white" if confusion[i, j] > confusion.max()/2. else "black",
            )
    return ax
