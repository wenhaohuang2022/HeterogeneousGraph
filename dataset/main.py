import dgl
import torch
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from net_to_het import generate_het
import os
import pickle
import numpy
import pandas


def process_line(line):
    parts = line.strip().split(',')
    str1, str2, str3 = parts[0].replace(' ', '_'), parts[1], parts[2]
    new_str = f"{str1}_da_{str2}_{str3}.scs"
    label_vector = torch.tensor([float(parts[i]) for i in range(3, 9)], dtype=torch.float32)
    return new_str, label_vector


def read_file_and_process(file_path):
    graphs = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            # 获取电路名称和标签
            netlist_file_name, label_vector = process_line(line)
            # 打开文件名为netlist_file_name的仿真网表文件
            script_dir = os.path.dirname(os.path.abspath(__file__))
            netlist_file_path = os.path.join(script_dir, 'raw_labels', netlist_file_name)
            # 根据网表生成异构图 hg
            hg = generate_het(netlist_file_path)
            graphs.append(hg)
            labels.append(label_vector)
    labels = torch.stack(labels)
    return graphs, labels


class HeteroGraphDataset(DGLDataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        super().__init__(name='hetero_graph_dataset')

    def process(self):
        pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump((dataset.graphs, dataset.labels), file)


def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        graphs, labels = pickle.load(file)
    return HeteroGraphDataset(graphs, labels)


file_path = 'amp_conclusion_ABC_KG_SMIC180.txt'  # 应该放在同一文件夹下
graphs, labels = read_file_and_process(file_path)
masks = torch.ones(len(graphs), dtype=torch.bool)  # 全部掩码设置为1 掩码暂未存入
dataset = HeteroGraphDataset(graphs, labels)
# 数据集建立

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'ckt_dataset.pkl')
save_dataset(dataset, save_path)
# 数据集保存

