import re
import dgl
import torch as th
import os

net_type = {'VINN': 1,
            'VINP': 2,
            'VOUT': 3,
            'IBIAS': 4,
            'VDD': 5,
            'VSS': 6}

def extract_col(matrix):
    # 提取第一列和第二列
    src = [row[0] for row in matrix]
    dst = [row[1] for row in matrix]
    return src, dst


def extract_subckt_blocks_from_string(input_str):
    subckt_blocks = []
    start_idx = input_str.find('subckt')

    while start_idx != -1:
        end_idx = input_str.find('end', start_idx)
        if end_idx == -1:
            break

        subckt_blocks.append(input_str[start_idx:end_idx + 3])
        start_idx = input_str.find('subckt', end_idx)

    return '\n'.join(subckt_blocks)


def add_reverse_edges(hetero_graph):
    new_edges = {}
    for e_type in hetero_graph.canonical_etypes:
        u, v = hetero_graph.edges(etype=e_type)
        new_edges[e_type] = (u, v)
        reverse_etype = (e_type[2], e_type[1] + '_reverse', e_type[0])
        new_edges[reverse_etype] = (v, u)
    return dgl.heterograph(new_edges)


def generate_het(file_path):
    # 读取网表文件
    with open(file_path, 'r') as file:
        data = file.read()
        data = extract_subckt_blocks_from_string(data)
        data = re.sub(r'``.*?``', '', data, flags=re.DOTALL)
    # 按照器件类型分组
    devices_data = re.split(r'(PM\d+|NM\d+|R\d+|C\d)', data)[1:]

    # 器件/网的节点数目
    num_pmos = num_nmos = num_r = num_c = num_net = 0

    # 初始化边
    edge_dp2n = []
    edge_gp2n = []
    edge_sp2n = []
    edge_bp2n = []
    edge_dn2n = []
    edge_gn2n = []
    edge_sn2n = []
    edge_bn2n = []
    edge_r2n = []
    edge_c2n = []

    # 键值对
    # key=在网表中，器件/网的名称；value=在异构图中，器件/网新分配的编号
    net_name = {}
    pmos_name = {}
    nmos_name = {}
    r_name = {}
    c_name = {}
    # 初始化节点特征
    pmos_feature = []
    nmos_feature = []
    r_feature = []
    c_feature = []
    net_feature=[]

    # 遍历每个器件数据
    for i in range(0, len(devices_data), 2):
        # 获取器件类型P,N,R,C
        device_type = devices_data[i][0].upper()
        # 获取器件参数数据,devices_data[i]是器件名称,[i+1]是名称后面剩余的信息
        device_data = devices_data[i + 1]
        # 初始化特征向量
        vector = []

        if device_type in ['P', 'N']:  # PMOS 或 NMOS
            # 使用正则表达式提取特征参数
            parameters = re.findall(r'(\w+)=([\d.]+(?:e-?\d+)?)', device_data)
            # 将提取的参数存入字典
            param_dict = {}
            for key, value_with_unit in parameters:
                value = float(value_with_unit)
                param_dict[key] = value
            if param_dict.get('m', 0):
                m = int(param_dict.get('m', 0))
            else:
                m = 1
            vector = [
                param_dict.get('w', 0),  # 注意！w是一个finger的宽度,总宽度还需乘以finger数。
                param_dict.get('l', 0),
                m
            ]
            # 提取连接关系
            if device_type == 'P':
                pmos_feature.append(vector)
                # 获得pmos的编号:pmos_name[pmos]==当前的num_pmos
                pmos = devices_data[i]
                pmos_name[pmos] = num_pmos
                num_pmos = num_pmos + 1

                initial_nets = device_data.split()[:4]
                nets = [re.sub(r'[()]', '', net) for net in initial_nets]
                # 获得net的编号
                net_index = []
                for net in nets:
                    if net not in net_name:
                        net_name[net] = num_net
                        if net in net_type.keys():
                            net_feature.append([net_type.get(net)])
                        else:
                            net_feature.append([0])
                        num_net = num_net + 1
                    net_index.append(net_name[net])


                # 添加边
                edge_dp2n.append([pmos_name[pmos], net_index[0]])
                edge_gp2n.append([pmos_name[pmos], net_index[1]])
                edge_sp2n.append([pmos_name[pmos], net_index[2]])
                edge_bp2n.append([pmos_name[pmos], net_index[3]])

            else:
                nmos_feature.append(vector)
                # 获得nmos的编号:nmos_name[nmos]==当前的num_nmos
                nmos = devices_data[i]
                nmos_name[nmos] = num_nmos
                num_nmos = num_nmos + 1
                # 提取相连的net
                initial_nets = device_data.split()[:4]
                nets = [re.sub(r'[()]', '', net) for net in initial_nets]
                # 获得net的编号
                net_index = []
                for net in nets:
                    if net not in net_name:
                        net_name[net] = num_net
                        if net in net_type.keys():
                            net_feature.append([net_type.get(net)])
                        else:
                            net_feature.append([0])
                        num_net = num_net + 1
                    net_index.append(net_name[net])
                # 添加边
                edge_dn2n.append([nmos_name[nmos], net_index[0]])
                edge_gn2n.append([nmos_name[nmos], net_index[1]])
                edge_sn2n.append([nmos_name[nmos], net_index[2]])
                edge_bn2n.append([nmos_name[nmos], net_index[3]])

        elif device_type == 'R':  # 电阻
            # 使用正则表达式提取电阻值
            resistance_value = float(re.findall(r'r=([\d.]+(?:e-?\d+)?)', device_data)[0])
            vector = [resistance_value]
            r_feature.append(vector)

            # 提取连接关系
            r = devices_data[i]
            r_name[r] = num_r
            num_r = num_r + 1
            # 提取相连的net
            initial_nets = device_data.split()[:2]
            nets = [re.sub(r'[()]', '', net) for net in initial_nets]
            # 获得net的编号
            net_index = []
            for net in nets:
                if net not in net_name:
                    net_name[net] = num_net
                    if net in net_type.keys():
                        net_feature.append([net_type.get(net)])
                    else:
                        net_feature.append([0])
                    num_net = num_net + 1
                net_index.append(net_name[net])
            # 添加边
            edge_r2n.append([r_name[r], net_index[0]])
            edge_r2n.append([r_name[r], net_index[1]])  # 这里假设了net_index[1]!=net_index[0]

        elif device_type == 'C':  # 电容
            # 使用正则表达式提取电容值
            capacitance_value = float(re.findall(r'c=([\d.]+(?:e-?\d+)?)', device_data)[0])
            vector = [capacitance_value]
            c_feature.append(vector)

            # 提取连接关系
            c = devices_data[i]
            c_name[c] = num_c
            num_c = num_c + 1
            # 提取相连的net
            initial_nets = device_data.split()[:2]
            nets = [re.sub(r'[()]', '', net) for net in initial_nets]
            # 获得net的编号
            net_index = []
            for net in nets:
                if net not in net_name:
                    net_name[net] = num_net
                    if net in net_type.keys():
                        net_feature.append([net_type.get(net)])
                    else:
                        net_feature.append([0])
                    num_net = num_net + 1
                net_index.append(net_name[net])
            # 添加边
            edge_c2n.append([c_name[c], net_index[0]])
            edge_c2n.append([c_name[c], net_index[1]])  # 这里假设了net_index[1]!=net_index[0]

    if num_c == 0:
        edge_c2n = [[0, 0], [0, 0]]
        c_feature = [[0.0]]
        num_c = 1

    if num_r == 0:
        edge_r2n = [[0, 0], [0, 0]]
        r_feature = [[0.0]]
        num_r = 1

    # 建立异构图
    # 建立连接关系
    graph_data = {
        ('P', 'dp2n', 'net'): (th.tensor(extract_col(edge_dp2n)[0]), th.tensor(extract_col(edge_dp2n)[1])),
        ('P', 'gp2n', 'net'): (th.tensor(extract_col(edge_gp2n)[0]), th.tensor(extract_col(edge_gp2n)[1])),
        ('P', 'sp2n', 'net'): (th.tensor(extract_col(edge_sp2n)[0]), th.tensor(extract_col(edge_sp2n)[1])),
        ('P', 'bp2n', 'net'): (th.tensor(extract_col(edge_bp2n)[0]), th.tensor(extract_col(edge_bp2n)[1])),
        ('N', 'dn2n', 'net'): (th.tensor(extract_col(edge_dn2n)[0]), th.tensor(extract_col(edge_dn2n)[1])),
        ('N', 'gn2n', 'net'): (th.tensor(extract_col(edge_gn2n)[0]), th.tensor(extract_col(edge_gn2n)[1])),
        ('N', 'sn2n', 'net'): (th.tensor(extract_col(edge_sn2n)[0]), th.tensor(extract_col(edge_sn2n)[1])),
        ('N', 'bn2n', 'net'): (th.tensor(extract_col(edge_bn2n)[0]), th.tensor(extract_col(edge_bn2n)[1])),
        ('R', 'r2n', 'net'): (th.tensor(extract_col(edge_r2n)[0]), th.tensor(extract_col(edge_r2n)[1])),
        ('C', 'c2n', 'net'): (th.tensor(extract_col(edge_c2n)[0]), th.tensor(extract_col(edge_c2n)[1]))
    }

    g = dgl.heterograph(graph_data)
    g = add_reverse_edges(g)  # 增加反向边，变成无向图。
    # 设置特征向量
    g.nodes['P'].data['x'] = th.tensor(pmos_feature)
    g.nodes['N'].data['x'] = th.tensor(nmos_feature)
    g.nodes['C'].data['x'] = th.tensor(c_feature)
    g.nodes['R'].data['x'] = th.tensor(r_feature)
    g.nodes['net'].data['x'] = th.tensor(net_feature)
    return g


if __name__=='__main__':
    # 用于测试generate_het函数
    script_dir = os.path.dirname(os.path.abspath(__file__))
    netlist_file_path = os.path.join(script_dir, 'raw_labels', 'AZC_da_1_500pF.scs')
    hg = generate_het(netlist_file_path)
    print(hg.nodes['net'].data['x'])
