import torch
from torch.utils.data import DataLoader, random_split
import dgl


def collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.stack(labels)
    return batched_graph, batched_labels


def dataset_split(dataset, train_ratio=0.7, validate_ratio=0.15, batch_size=32, collate_fn=collate_fn):
    num_data = len(dataset)
    num_train = int(train_ratio * num_data)
    num_validate = int(validate_ratio * num_data)
    num_test = num_data - num_train - num_validate

    # 使用 random_split 函数划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_validate, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # 划分完成
    return train_loader, val_loader, test_loader
