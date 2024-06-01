import dgl
import torch
import os
from torch.utils.data import DataLoader
from main import load_dataset, save_path
from visualization import plot_heterograph
def collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.stack(labels)
    return batched_graph, batched_labels

loaded_dataset = load_dataset(save_path)
dataloader = DataLoader(loaded_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
for batched_g, batched_l in dataloader:
    print(batched_g)
    print(batched_l)
    break

output_dir = 'graph_images'
for i in range(0,307):
    graph_example,label_example = dataloader.dataset.__getitem__(i)
    plot_heterograph(graph_example,num=i,plot=False,save=True)



