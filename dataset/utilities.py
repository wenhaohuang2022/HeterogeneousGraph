import dgl
import torch
import os
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import pickle
from models import HeteroGraphDataset
import pandas
import  numpy

# 数据集保存和加载
def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump((dataset.graphs, dataset.labels,dataset.masks), file)

def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        g, l, m = pickle.load(file)
    return HeteroGraphDataset(g,l,m)

# 数据集划分
def collate_fn(batch):
    graphs, labels,masks = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.stack(labels)
    batched_masks = torch.stack(masks)
    return batched_graph, batched_labels,batched_masks


def dataset_split(dataset, train_ratio=0.7, validate_ratio=0.15, batch_size=32, collate_fn=collate_fn):
    num_data = len(dataset)
    num_train = int(train_ratio * num_data)
    num_validate = int(validate_ratio * num_data)
    num_test = num_data - num_train - num_validate

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_validate, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # 划分完成
    return train_loader, val_loader, test_loader


# 数据预处理：节点特征标准化
def normalization(raw_features):
    raw_features_np = raw_features.numpy()
    # 创建 StandardScaler 对象
    scaler = StandardScaler()
    # 对特征进行标准化
    standard_scaled_features_np = scaler.fit_transform(raw_features_np)
    # 将标准化后的特征转换回 torch.tensor
    standard_scaled_features = torch.tensor(standard_scaled_features_np, dtype=torch.float32)
    return standard_scaled_features


# 数据可视化
def plot_heterograph(g, num, plot=True, save=False):
    nx_g = g.to_networkx()
    # 给不同类型的节点赋予不同颜色
    color_map = {
        'net': '#6C30A2',
        'C': '#007E6D',
        'R': '#FFB749',
        'P': '#0043AC',
        'N': '#D82D5B'
    }
    # 获取节点类型并为每个节点分配颜色
    node_colors = []
    for n, d in nx_g.nodes(data=True):
        ntype = d['ntype']
        node_colors.append(color_map[ntype])
    # 绘制图形
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(nx_g)  # 使用 spring layout 布局
    nx.draw(nx_g, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=1, font_color="white",
            edge_color="#000000")
    graph_name = f'graph visualization No.{num}'
    plt.title(graph_name)
    if plot:
        plt.show()
    if save:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_save_path = os.path.join(script_dir, 'graph_images', f'{graph_name}.png')
        plt.savefig(image_save_path)
        plt.close()
    return
