import torch
from dgl.data import DGLDataset

class HeteroGraphDataset(DGLDataset):
    def __init__(self, graphs, labels, masks):
        self.graphs = graphs
        self.labels = labels
        self.masks = masks
        super().__init__(name='hetero_graph_dataset')

    def process(self):
        pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.masks[idx]

    def __len__(self):
        return len(self.graphs)
