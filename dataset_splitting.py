import torch
from torch.utils.data import DataLoader, random_split
import dgl
from main import dataset

# 定义划分的比例
train_ratio = 0.7
validate_ratio = 0.15
test_ratio = 0.15
# 计算划分的样本数量
num_data = len(dataset)
num_train = int(train_ratio * num_data)
num_validate = int(validate_ratio * num_data)
num_test = num_data - num_train - num_validate

# 使用 random_split 函数划分数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_validate, num_test])

# 创建数据加载器
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 划分完成
