from utilities import *

# demo 1 加载数据集，划分成训练集、验证集、测试集
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'CktDataset.pkl')
loaded_dataset = load_dataset(dataset_path)

train_loader, validate_loader, test_loader = dataset_split(loaded_dataset, batch_size=8, collate_fn=collate_fn)

for batched_g, batched_l, batched_m in train_loader:
    print('train dataset:')
    print('batched_graph:\n',batched_g)
    print('batched_label:\n',batched_l)
    print('batched_mask:\n',batched_m)
    break
for batched_g, batched_l, batched_m in validate_loader:
    print('validate dataset:')
    print('batched_graph:\n',batched_g)
    print('batched_label:\n',batched_l)
    print('batched_mask:\n',batched_m)
    break
for batched_g, batched_l, batched_m in test_loader:
    print('test dataset:')
    print('batched_graph:\n',batched_g)
    print('batched_label:\n',batched_l)
    print('batched_mask:\n',batched_m)
    break

# demo 2 可视化异构图，并存入文件夹（可选）
dataloader = DataLoader(loaded_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
for i in [0, 100, 300]:
    graph_example, label_example, mask_example = dataloader.dataset.__getitem__(i)
    plot_heterograph(graph_example, num=i, plot=True, save=False)

# demo 3 节点特征标准化
for batched_g, batched_l, batched_mask in train_loader:
    print('标准化前：\n',batched_g.nodes['P'].data['x'] )
    for ntype in batched_g.ntypes:
        batched_g.nodes[ntype].data['x'] = normalization(batched_g.nodes[ntype].data['x'])
    print('标准化后：\n',batched_g.nodes['P'].data['x'] )
    break
