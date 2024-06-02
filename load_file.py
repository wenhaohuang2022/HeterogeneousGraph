import dgl
import torch
import os
from torch.utils.data import DataLoader
from main import load_dataset, save_path
from visualization import plot_heterograph
from dataset_splitting import dataset_split, collate_fn

loaded_dataset = load_dataset(save_path)
train_loader, validate_loader, test_loader = dataset_split(loaded_dataset, collate_fn=collate_fn)

for batched_g, batched_l in train_loader:
    print('train dataset:')
    print(batched_g)
    print(batched_l)
    break
for batched_g, batched_l in validate_loader:
    print('validate dataset:')
    print(batched_g)
    print(batched_l)
    break
for batched_g, batched_l in test_loader:
    print('test dataset:')
    print(batched_g)
    print(batched_l)
    break

'''visualizing the graphs'''
# output_dir = 'graph_images'
dataloader = DataLoader(loaded_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
for i in range(0, 2):
    graph_example, label_example = dataloader.dataset.__getitem__(i)
    plot_heterograph(graph_example, num=i, plot=True, save=False)
