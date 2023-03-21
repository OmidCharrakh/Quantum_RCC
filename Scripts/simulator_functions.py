import pandas as pd
import numpy as np
import networkx as nx
import torch as th
from torch.utils.data import DataLoader

import distance_functions as dFun
import utilities_functions as uFun
###########################################################
# channel simulator
###########################################################


class NN_block(th.nn.Module):
    def __init__(self, n_nodes, dim_noise=1):
        super(NN_block, self).__init__()

        dim_in = dim_noise + 3*(n_nodes-1)
        dim_out = 3
        layers = []
        layers.append(th.nn.Linear(dim_in, 20))
        layers.append(th.nn.ReLU())
        # layers.append(th.nn.Linear(20, 20));layers.append(th.nn.ReLU())
        # layers.append(th.nn.Linear(200, 100))
        # layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(20, dim_out))
        self.layers = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class nn_simulator(th.nn.Module):
    def __init__(self,
                 n_nodes=8,
                 batch_size=50,
                 n_epochs=10,
                 lr=1e-3,
                 patience=5,
                 dim_noise=10,
                 model_path=None,
                 ):
        super(nn_simulator, self).__init__()

        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.patience = patience
        self.dim_noise = dim_noise
        self.criterion = dFun.MMDloss(batch_size)  # nn.MSELoss()
        self.model_path = model_path
        self.blocks_list = th.nn.ModuleList()
        for index_node in range(self.n_nodes):
            block = NN_block(self.n_nodes, self.dim_noise)
            self.blocks_list.append(block)

    def generate_noise(self, gen_size):
        noise_dict = {n: th.zeros(gen_size, self.dim_noise).normal_(0, 1) for n in range(self.n_nodes)}
        return noise_dict

    def forward(self, gen_size, adjacency_matrix):
        topological_order = [node for node in nx.topological_sort(nx.DiGraph(adjacency_matrix.numpy()))]
        gen_data = {n: None for n in range(self.n_nodes)}
        local_noises = self.generate_noise(gen_size)
        for i in topological_order:
            local_noise = [local_noises[i]]
            parent_data = []
            for j in topological_order:
                if j != i:
                    if adjacency_matrix[j, i]:
                        parent_data.append(gen_data[j])
                    else:
                        parent_data.append(th.zeros(gen_size, 3))
            input_data = th.cat([v for c in [local_noise, parent_data] for v in c], axis=1)
            gen_data[i] = self.blocks_list[i](input_data)
        gen_data = [gen_data[n] for n in range(self.n_nodes)]
        return th.cat(gen_data, axis=1)

    def fit(self, data_tr, data_va, saving_path=None):
        dataset_tr, adjacencies_tr = data_tr
        dataset_va, adjacencies_va = data_va
        adjacencies_tr = adjacencies_tr.type(th.int64)
        adjacencies_va = adjacencies_va.type(th.int64)
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        early_stopping = uFun.EarlyStopping(patience=self.patience, checkpoint_path=self.model_path, minimize_score=True)
        prog_tr = []
        prog_va = []
        for epoch in range(self.n_epochs):
            loss_tr = []
            for dat_tr, adj_tr in zip(dataset_tr, adjacencies_tr):
                optimizer.zero_grad()
                loss = th.tensor(0, dtype=th.float32)
                for da_tr in DataLoader(dat_tr, batch_size=self.batch_size, shuffle=True, drop_last=True):
                    da_gen = self.forward(da_tr.shape[0], adj_tr)
                    loss += self.criterion(da_gen, da_tr)
                loss.backward()
                loss_tr.append(loss.item())
                optimizer.step()
            prog_tr.append(np.mean(loss_tr))
            loss_va = []
            with th.no_grad():
                for dat_va, adj_va in zip(dataset_va, adjacencies_va):
                    loss = th.tensor(0, dtype=th.float32)
                    for da_va in DataLoader(dat_va, batch_size=self.batch_size, shuffle=True, drop_last=True):
                        da_gen = self.forward(da_va.shape[0], adj_va)
                        loss += self.criterion(da_gen, da_va)
                    loss_va.append(loss.item())
                prog_va.append(np.mean(loss_va))
            if epoch % 1 == 0:
                print(f'Epoch: {epoch}: train_loss: {100*np.mean(loss_tr)}, valid_loss: {100*np.mean(loss_va)}')
            early_stopping(np.mean(loss_va), self)
            if early_stopping.early_stop:
                print('EARLY-STOPPING')
                break
        th.save(self.state_dict(), saving_path)
        df_prog = pd.DataFrame()
        df_prog['epoch'] = list(range(len(prog_tr)))
        df_prog['loss_tr'] = prog_tr
        df_prog['loss_va'] = prog_va
        return df_prog

    def reset_parameters(self):
        for block in self.blocks_list:
            if block is not None:
                block.reset_parameters()

    def re_load(self, model_path):
        if model_path is not None:
            self.load_state_dict(th.load(model_path))
            self.eval()
        else:
            raise ValueError('No fitted model found on the given path')
        return self

    def fit_ijk(self, data_tr, data_va, list_ijk, saving_path=None):
        dataset_tr, adjacencies_tr = data_tr
        dataset_va, adjacencies_va = data_va
        adjacencies_tr = adjacencies_tr.type(th.int64)
        adjacencies_va = adjacencies_va.type(th.int64)
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        early_stopping = uFun.EarlyStopping(patience=self.patience, checkpoint_path=self.model_path, minimize_score=True)
        prog_tr = []
        prog_va = []
        for epoch in range(self.n_epochs):
            loss_tr = []
            for dat_tr, A_base, (i, j, k) in zip(dataset_tr, adjacencies_tr, list_ijk):
                adj_tr = uFun.embed_graph(i, j, k, A_base, self.n_nodes)
                optimizer.zero_grad()
                loss = th.tensor(0, dtype=th.float32)
                for da_tr in DataLoader(dat_tr, batch_size=self.batch_size, shuffle=True, drop_last=True):
                    da_gen = self.forward(da_tr.shape[0], adj_tr)
                    da_gen = th.concat((da_gen[:, 3*(i):3*(i+1)], da_gen[:, 3*(j):3*(j+1)], da_gen[:, 3*(k):3*(k+1)]), dim=1)
                    loss += self.criterion(da_gen, da_tr)
                loss.backward()
                loss_tr.append(loss.item())
                optimizer.step()
            prog_tr.append(np.mean(loss_tr))
            loss_va = []
            with th.no_grad():
                for dat_va, adj_va in zip(dataset_va, adjacencies_va):
                    loss = th.tensor(0, dtype=th.float32)
                    for da_va in DataLoader(dat_va, batch_size=self.batch_size, shuffle=True, drop_last=True):
                        da_gen = self.forward(da_va.shape[0], adj_va)
                        loss += self.criterion(da_gen, da_va)
                    loss_va.append(loss.item())
                prog_va.append(np.mean(loss_va))
            if epoch % 1 == 0:
                print(f'epoch: {epoch}: train_loss: {100*np.mean(loss_tr)}, valid_loss: {100*np.mean(loss_va)}')
            early_stopping(np.mean(loss_va), self)
            if early_stopping.early_stop:
                print('EARLY-STOPPING')
                break
        th.save(self.state_dict(), saving_path)
        df_prog = pd.DataFrame()
        df_prog['epoch'] = list(range(len(prog_tr)))
        df_prog['loss_tr'] = prog_tr
        df_prog['loss_va'] = prog_va
        return df_prog
