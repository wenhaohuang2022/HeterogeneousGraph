# HeterogeneousGraph
Use heterogeneous graph neural networks to generate analog integrated circuit 
## 数据准备 dataset文件夹
- SMIC180_ckt文件夹内存有大量的仿真网表，从中整理出了307张非重复的网表，存放在raw_labels文件夹内<br>
- amp_conclusion_ABC_KG_SMIC180.txt文件中存有307张网表的仿真性能指标，作为数据集的标签<br>

- main.py：建立并保存数据集的主文件<br>
- net_to_het.py：读取网表，将其转化为异构图表示<br>
- utilities.py：工具类函数存放区<br>
- models.py：数据集模型、神经网络模型等<br>
- demo.py：检验某些工具的功能，如加载数据集、可视化异构图和标准化节点特征<br>

- CktDataset.pkl为最终的异构图数据集，含有307张图及其标签<br>
- graph_images文件夹内存有307张异构图的可视化图像，对应于307张网表<br>
## 模型搭建
