import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import math


def federated_learning_bottleneck(n, model_size=1):
    """
    计算联邦学习中的通信瓶颈带宽（中央服务器）。
    """
    # 每轮通信中，服务器发送和接收的总通信量
    total_communication = 2 * n * model_size
    return total_communication


def distributed_learning_bottleneck(n, model_size=1):
    """
    计算端对端分布式学习中的通信瓶颈带宽（度数最大的节点）。
    """
    # 生成随机图
    p = min(1, np.log(n) / n)  # 确保图连通
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
    degrees = [degree for node, degree in G.degree()]
    max_degree = max(degrees)
    bottleneck_bandwidth = max_degree * model_size
    return bottleneck_bandwidth


def cluster_learning_bottleneck(n, model_size=1, cluster_size=None, min_clusters_per_node=1):
    """
    计算聚类分布式学习中的通信瓶颈带宽。

    参数：
    - n: 节点数量
    - model_size: 模型参数大小
    - cluster_size: 团的大小，如果为 None，则随机生成
    - min_clusters_per_node: 每个节点至少所属的团数量
    """
    nodes = list(range(n))
    clusters = []
    member_clusters = {i: [] for i in nodes}

    # 计算默认的团大小
    if cluster_size is None:
        cluster_size = random.randint(2, n)

    # 确定团的数量
    num_clusters = max(n // cluster_size, 1)

    # 生成团
    for _ in range(num_clusters):
        cluster_members = random.sample(nodes, cluster_size)
        clusters.append(cluster_members)
        for member in cluster_members:
            member_clusters[member].append(len(clusters) - 1)

    # 确保每个节点至少在 min_clusters_per_node 个团中
    for node in nodes:
        while len(member_clusters[node]) < min_clusters_per_node:
            # 随机选择一个团，加入该节点
            cluster_id = random.randint(0, num_clusters - 1)
            if node not in clusters[cluster_id]:
                clusters[cluster_id].append(node)
                member_clusters[node].append(cluster_id)

    # 计算每个节点的通信量
    node_communication = [0] * n
    for node in nodes:
        clusters_of_node = member_clusters[node]
        for cluster_id in clusters_of_node:
            cluster_members = clusters[cluster_id]
            k = len(cluster_members)
            # 安全多方计算的通信量，假设需要与其他所有成员交换数据
            node_communication[node] += (k - 1) * model_size  # 与团内其他成员通信

    bottleneck_bandwidth = max(node_communication)
    return bottleneck_bandwidth


def virtual_nodes_bottleneck(n, model_size=1, vns_per_node=16):
    """
    计算虚拟节点方法中的通信瓶颈带宽。

    参数：
    - n: 真实节点数量
    - model_size: 模型参数大小
    - vns_per_node: 每个真实节点的虚拟节点数量
    """
    # 总的虚拟节点数量
    total_vns = n * vns_per_node

    # 生成虚拟节点之间的随机拓扑
    p = min(1, np.log(total_vns) / total_vns)  # 确保图连通
    G = nx.erdos_renyi_graph(total_vns, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(total_vns, p)

    # 计算虚拟节点之间的最大度数
    degrees = [degree for node, degree in G.degree()]
    max_degree = max(degrees)

    # 计算通信瓶颈带宽
    # 每个真实节点与其虚拟节点通信 + 虚拟节点之间的通信 + 虚拟节点返回真实节点的通信
    bottleneck_bandwidth = vns_per_node * model_size + max_degree * model_size + vns_per_node * model_size

    return bottleneck_bandwidth


def admm_learning_bottleneck(n, model_size=1, iterations=2):
    """
    计算基于 ADMM 的去中心化学习的通信瓶颈带宽。

    参数：
    - n: 节点数量
    - model_size: 模型参数大小
    - iterations: 每轮 ADMM 迭代次数
    """
    # 生成随机图，确保连通
    p = min(1, np.log(n) / n)
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)

    # 计算每个节点的度数
    degrees = [degree for node, degree in G.degree()]
    max_degree = max(degrees)

    # 每次迭代中，节点需要与其所有邻居通信，迭代次数为 iterations
    bottleneck_bandwidth = max_degree * model_size * iterations
    return bottleneck_bandwidth


def simulate_bottleneck_bandwidth(n_values, model_size=1):
    federated_bandwidth = []
    distributed_bandwidth = []
    cluster_bandwidth = []
    cluster_sqrt_bandwidth = []
    cluster_sqrt_many_clusters_bandwidth = []
    virtual_nodes_bandwidth = []
    admm_bandwidth = []

    for n in n_values:
        print(f"Simulating for n = {n}")
        # 联邦学习
        fl_bandwidth = federated_learning_bottleneck(n, model_size)
        federated_bandwidth.append(fl_bandwidth)

        # 端对端分布式学习
        dl_bandwidth = distributed_learning_bottleneck(n, model_size)
        distributed_bandwidth.append(dl_bandwidth)

        # 聚类分布式学习（随机拓扑）
        cl_bandwidth = cluster_learning_bottleneck(n, model_size)
        cluster_bandwidth.append(cl_bandwidth)

        # 聚类分布式学习（团大小固定为 7）
        sqrt_cluster_size = 7
        cl_sqrt_bandwidth = cluster_learning_bottleneck(n, model_size, cluster_size=sqrt_cluster_size)
        cluster_sqrt_bandwidth.append(cl_sqrt_bandwidth)

        # 聚类分布式学习（团大小固定为 7，每个节点至少在 n/30 个团中）
        min_clusters_per_node = max(1, n // 30)
        cl_sqrt_many_bandwidth = cluster_learning_bottleneck(n, model_size, cluster_size=sqrt_cluster_size,
                                                             min_clusters_per_node=min_clusters_per_node)
        cluster_sqrt_many_clusters_bandwidth.append(cl_sqrt_many_bandwidth)

        # 虚拟节点方法
        vns_per_node = 3
        vn_bandwidth = virtual_nodes_bottleneck(n, model_size, vns_per_node)
        virtual_nodes_bandwidth.append(vn_bandwidth)

        # ADMM 方法
        admm_iter = 2  # 假设每轮有 2 次迭代
        admm_bw = admm_learning_bottleneck(n, model_size, iterations=admm_iter)
        admm_bandwidth.append(admm_bw)

    return {
        'Federated Learning': federated_bandwidth,
        'Distributed Learning': distributed_bandwidth,
        'Cluster Learning (Random)': cluster_bandwidth,
        'Cluster Learning (Fixed sqrt(n) size)': cluster_sqrt_bandwidth,
        'Cluster Learning (Fixed sqrt(n) size, Many Clusters)': cluster_sqrt_many_clusters_bandwidth,
        'Virtual Nodes Method': virtual_nodes_bandwidth,
        'ADMM Method': admm_bandwidth
    }


def plot_bottleneck_bandwidth(n_values, bandwidth_dict):
    plt.figure(figsize=(12, 8))

    for method, bandwidth in bandwidth_dict.items():
        plt.plot(n_values, bandwidth, label=method)

    plt.xlabel('Number of Nodes (n)', fontsize=14)
    plt.ylabel('Bottleneck Bandwidth (units of model size)', fontsize=14)
    plt.title('Communication Bottleneck Bandwidth vs. Number of Nodes', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('communication_bottleneck_bandwidth.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    n_values = list(range(10, 201, 5))  # 从10到200，步长为5
    model_size = 1  # 假设模型参数大小为1，关注相对通信量

    bandwidth_dict = simulate_bottleneck_bandwidth(n_values, model_size)
    plot_bottleneck_bandwidth(n_values, bandwidth_dict)