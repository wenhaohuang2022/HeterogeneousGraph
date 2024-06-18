import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import os
from data.utils import *

epsilon = 1
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'CktDataset.pkl')
loaded_dataset = load_dataset(dataset_path)
batch_size = 32
train_loader, validate_loader, test_loader = dataset_split(loaded_dataset, batch_size=batch_size,
                                                           collate_fn=collate_fn)

canonical_etypes = [
    ('P', 'dp2n', 'net'),
    ('net', 'dp2n_reverse', 'P'),
    ('P', 'gp2n', 'net'),
    ('net', 'gp2n_reverse', 'P'),
    ('P', 'sp2n', 'net'),
    ('net', 'sp2n_reverse', 'P'),
    ('P', 'bp2n', 'net'),
    ('net', 'bp2n_reverse', 'P'),
    ('N', 'dn2n', 'net'),
    ('net', 'dn2n_reverse', 'N'),
    ('N', 'gn2n', 'net'),
    ('net', 'gn2n_reverse', 'N'),
    ('N', 'sn2n', 'net'),
    ('net', 'sn2n_reverse', 'N'),
    ('N', 'bn2n', 'net'),
    ('net', 'bn2n_reverse', 'N'),
    ('R', 'r2n', 'net'),
    ('net', 'r2n_reverse', 'R'),
    ('C', 'c2n', 'net'),
    ('net', 'c2n_reverse', 'C')
]
in_dims = {
    'P': 3,
    'net': 1,
    'N': 3,
    'R': 1,
    'C': 1
}
node_types = ['P', 'net', 'N', 'R', 'C']


class HeteroGraphNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        # in_dim 选为3，hidden_dim选为100，out_dim为6。
        super(HeteroGraphNN, self).__init__()
        # 线性层，统一特征向量的维数到in_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear_transform = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], in_dim) for ntype in node_types
        })
        # 第一层卷积
        self.conv1 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(in_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第二次卷积
        self.conv2 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv3 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv4 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv5 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv6 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv7 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv8 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv9 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 第三次卷积
        self.conv10 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_dim, hidden_dim) for etype in canonical_etypes},
            aggregate='sum')
        # 输出目标层
        self.final_linear = None

    def forward(self, g, inputs):
        # g为异构图对象，inputs为g各个节点的特征矩阵
        # 特征维度变换
        h = {ntype: self.linear_transform[ntype](feat) for ntype, feat in inputs.items()}

        # 第一层卷积
        h = self.conv1(g, h)
        h = {k: torch.relu(v) for k, v in h.items()}

        # 第二层卷积
        h = self.conv2(g, h)
        h = {k: torch.relu(v) for k, v in h.items()}

        # 第三层卷积
        h = self.conv3(g, h)
        h = {k: torch.relu(v) for k, v in h.items()}

        # 将特征向量拼接在一起
        h_concat = torch.cat([h[ntype].view(-1) for ntype in node_types], dim=0)
        total_nodes = len(h_concat)
        # 动态创建 final_linear层
        if self.final_linear is None or self.final_linear.in_features != total_nodes * self.hidden_dim:
            self.final_linear = nn.Linear(total_nodes, self.out_dim)

        output = self.final_linear(h_concat)

        return output


import torch.optim as optim

model = HeteroGraphNN(in_dim=3, hidden_dim=100, out_dim=6)
epochs = 3
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
acc_record = []
loss_record = []
for epoch in range(epochs):

    model.train()
    for batched_g, batched_l, batched_m in train_loader:
        # 标准化
        for ntype in batched_g.ntypes:
            batched_g.nodes[ntype].data['x'] = normalization(batched_g.nodes[ntype].data['x'])
        batched_l = normalization(batched_l)
        un_batched_g = dgl.unbatch(batched_g)
        for i in range(0, batch_size):
            if i < len(un_batched_g):
                optimizer.zero_grad()
                g = un_batched_g[i]
                inputs = {ntype: g.nodes[ntype].data['x'] for ntype in node_types}
                # input是字典，键值对为：节点类型-节点特征矩阵
                l = batched_l[i]
                output = model(g, inputs)
                loss = loss_fn(output, l)
                loss.backward()
                optimizer.step()
            else:
                pass

    print(f'training of epoch {epoch} is done')

    model.eval()
    num_correct = 0
    num_examples = 0
    loss_sum = 0

    for batched_g, batched_l, batched_m in validate_loader:
        for ntype in batched_g.ntypes:
            batched_g.nodes[ntype].data['x'] = normalization(batched_g.nodes[ntype].data['x'])
        batched_l = normalization(batched_l)
        un_batched_g = dgl.unbatch(batched_g)

        for i in range(0, batch_size):
            if i < len(un_batched_g):
                num_examples += 1
                g = un_batched_g[i]
                inputs = {ntype: g.nodes[ntype].data['x'] for ntype in node_types}  # input是字典，键值对为：节点类型-节点特征矩阵
                l = batched_l[i]
                output = model(g, inputs)

                loss = loss_fn(output, l)
                loss_sum += loss.detach().item()
                if loss < epsilon:
                    num_correct += 1
                else:
                    pass
            else:
                pass

    print(f'validation of epoch {epoch} is done')
    acc = num_correct / num_examples
    loss_average = loss_sum / num_examples
    print(f'accuracy is  = {acc * 100}%')
    print(f'loss average is {loss_average}')
    acc_record.append(acc)
    loss_record.append(loss_average)

print(loss_record)
print(acc_record)

import matplotlib.pyplot as plt
plt.plot(list(range(1, len(loss_record)+1)), loss_record, marker='o')  # 绘制折线图，使用圆形标记点
plt.xlabel('x')  # x轴标签
plt.ylabel('loss')  # y轴标签
plt.title('Loss Plot')  # 图表标题
plt.grid(True)  # 显示网格
plt.show()  # 显示图表'''

plt.plot(list(range(1, epochs + 1)), acc_record, marker='*',color='orange')  # 绘制折线图，使用圆形标记点
plt.xlabel('epochs')  # x轴标签
plt.ylabel('acc')  # y轴标签
plt.title('Acc Plot')  # 图表标题
plt.grid(True)  # 显示网格
plt.show()  # 显示图表'''

