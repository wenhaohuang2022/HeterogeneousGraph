import networkx as nx
import  matplotlib.pyplot as plt
import  os
def plot_heterograph(g,num,plot = True,save=False):
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
    graph_name=f'graph visualization No.{num}'
    plt.title(graph_name)
    if plot:
        plt.show()
    if save:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_save_path = os.path.join(script_dir, 'graph_images', f'{graph_name}.png')
        plt.savefig(image_save_path)
        plt.close()
    return



