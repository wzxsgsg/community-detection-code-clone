import community
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import javalang
from javalang.ast import Node
import os
from anytree import AnyNode, RenderTree
from create_ast_graph import create_ast_graph_func
import numpy as np

def louvain_detect(g, x):
    partition = community_louvain.best_partition(g)  # 社区检测
    pvq = list(set(partition.values()))
    #     print("pvq:", pvq)
    #     print("x_length:", len(x))
    val_dict = {}  # 每个分区所对应的token
    val_dict_num = {}  # 每个分区所对应token的序号
    for key in partition.keys():  # 求每个区间所含有的token
        for i in pvq:
            if partition[key] == i:
                if i in val_dict.keys():
                    val_dict[i].append(x[key])
                    val_dict_num[i].append(key)
                else:
                    val_dict[i] = [x[key]]
                    val_dict_num[i] = [key]

    return partition, val_dict, val_dict_num

def get_subgraph_centrality(partition, g):
    G_new = nx.Graph()
    for u, v in g.edges():
        u_part = partition[u]
        v_part = partition[v]
        if u_part != v_part and not G_new.has_edge(u_part, v_part):
            G_new.add_edge(u_part, v_part)

    subgraph_degree_centrality = list(nx.degree_centrality(G_new).values())
    subgraph_betweenness_centrality = list(nx.betweenness_centrality(G_new).values())
    subgraph_closeness_centrality = list(nx.closeness_centrality(G_new).values())
    subgraph_harmonic_centrality = list(nx.harmonic_centrality(G_new).values())
    subgraph_centrality_result = ((np.array(subgraph_degree_centrality)+np.array(subgraph_betweenness_centrality)+np.array(subgraph_closeness_centrality)+np.array(subgraph_harmonic_centrality))/4).tolist()
    
    subgraph_result = dict(zip(G_new.nodes(), subgraph_centrality_result))
    return subgraph_result

# partition, val_dict, val_dict_num = louvain_detect(g, x)


# 得到每个分区的子图，以及子图的结点和边
def get_subgraph(partition, g):
    subgraphs = {}
    for node, comm in partition.items():
        if comm not in subgraphs:
            subgraphs[comm] = nx.Graph()
        subgraphs[comm].add_node(node)

    for u, v in g.edges():
        if partition[u] == partition[v]:
            subgraphs[partition[u]].add_edge(u, v)

    #     for i, subgraph in enumerate(subgraphs.values()):
    #         print(f"Subgraph {i+1} degree centrality: {sum(dict(nx.degree(subgraph)).values())}")
    #         print(nx.degree_centrality(subgraph))
    #         print("Nodes:", list(subgraph.nodes()))
    #         print("Edges:", list(subgraph.edges()))

    return subgraphs


def compute_centrality(subgraphs, subgraph_result):
    results = {}
    for i, subgraph_dict in enumerate(subgraphs.items()):
        subgraph_key = subgraph_dict[0]
        subgraph = subgraph_dict[1]
        nodes = list(subgraph.nodes())
        edges = list(subgraph.edges())
        
        # 子图中心性分析计算
        degree_centrality = sum(dict(nx.degree(subgraph)).values()) / len(nodes)
        betweenness_centrality = sum(nx.betweenness_centrality(subgraph).values()) / len(nodes)
        closeness_centrality = sum(nx.closeness_centrality(subgraph).values()) / len(nodes)
        harmonic_centrality = sum(nx.harmonic_centrality(subgraph).values()) / len(nodes)

        # 子图结点中心性分析计算
        node_degree_centrality = list(nx.degree_centrality(subgraph).values())
        node_betweenness_centrality = list(nx.betweenness_centrality(subgraph).values())
        node_closeness_centrality = list(nx.closeness_centrality(subgraph).values())
        node_harmonic_centrality = list(nx.harmonic_centrality(subgraph).values())

        node_subgraph_results = []

        node_subgraph_results = ((np.array(node_degree_centrality)+np.array(node_betweenness_centrality)+np.array(node_closeness_centrality)+np.array(node_harmonic_centrality))/4).tolist()
        subgraph_average_results = (degree_centrality + betweenness_centrality + closeness_centrality + harmonic_centrality) / 4
        
        # result[0]代表子图中心性，此处子图中心性计算方法为：（结点求和平均+看作结点通过networkx计算结点中心性）/ 2
#         results[f"Subgraph {subgraph_key} centrality"] = [(subgraph_result[subgraph_key]+subgraph_average_results) / 2] 
        
        # 看作结点通过networkx计算结点中心性
        results[f"Subgraph {subgraph_key} centrality"] = [subgraph_result[subgraph_key]]
        
        results[f"Subgraph {subgraph_key} centrality"].append(node_subgraph_results)  # result[1] 代表整个社区子图的所有结点的中心性计算结果

    mutiply_centrality = {}
    for key in results.keys():
#         mutiply_centrality[key] = [m * results[key][1] for m in results[key][2]]
        mutiply_centrality[key] = (np.array(results[key][1])*results[key][0]).tolist()
    # avg_centrality = sum(results) / len(results)
    # print(f"Average Centrality: {avg_centrality}")

    return results, mutiply_centrality


def get_whole_graph_centrality(g):
    whole_graph_degree_centrality = list(nx.degree_centrality(g).values())
    #     whole_graph_katz_centrality = list(nx.katz_centrality(g, alpha=0.1, beta=1.0, max_iter=10000).values())
    whole_graph_betweenness_centrality = list(nx.betweenness_centrality(g).values())
    whole_graph_closeness_centrality = list(nx.closeness_centrality(g).values())
    whole_graph_harmonic_centrality = list(nx.harmonic_centrality(g).values())

    whole_graph_results = []
#     for j in range(len(whole_graph_degree_centrality)):
#         #         whole_graph_results.append((whole_graph_degree_centrality[j] + whole_graph_katz_centrality[j] + whole_graph_betweenness_centrality[j] + whole_graph_closeness_centrality[j] + whole_graph_harmonic_centrality[j]) / 5)
#         whole_graph_results.append((whole_graph_degree_centrality[j] + whole_graph_betweenness_centrality[j] +
#                                     whole_graph_closeness_centrality[j] + whole_graph_harmonic_centrality[j]) / 4)
    whole_graph_results = ((np.array(whole_graph_degree_centrality)+np.array(whole_graph_betweenness_centrality)+np.array(whole_graph_closeness_centrality)+np.array(whole_graph_harmonic_centrality))/4).tolist()
    #     print(whole_graph_results)
    return whole_graph_results

def final_centrality(subgraphs, mutiply_centrality, whole_graph_results):
    final_centrality_result = []
    final_centrality_result = whole_graph_results
    for i, subgraph_dict in enumerate(subgraphs.items()):
        subgraph_key = subgraph_dict[0]
        subgraph = subgraph_dict[1]
        nodes = list(subgraph.nodes())
        Subgraph_centrality = mutiply_centrality[f"Subgraph {subgraph_key} centrality"]
        for j in range(len(Subgraph_centrality)):
#             final_centrality_result[nodes[j]] = final_centrality_result[nodes[j]] * Subgraph_centrality[j] #3权重相乘
            final_centrality_result[nodes[j]] = (final_centrality_result[nodes[j]] + Subgraph_centrality[j]) / 2 #子图中心性*子图结点中心性与全局中心性做平均
#             final_centrality_result[nodes[j]] = Subgraph_centrality[j] # 只使用子图及其结点中心性

    return final_centrality_result


# 计算所有token的权重，相同的token中心性相加
# print(x)
def get_token_total_weight(x, final_centrality_result):
    token_weight_dict = {}
    for i in range(len(x)):
        if x[i] in token_weight_dict.keys():
            token_weight_dict[x[i]] = token_weight_dict[x[i]] + final_centrality_result[i]
        else:
            token_weight_dict[x[i]] = final_centrality_result[i]

    return token_weight_dict


# word2Vec嵌入与权重相乘

from gensim.models import Word2Vec
def word2vec_embedding(model, token_weight_dict):
    tokens = list(token_weight_dict.keys())
#     model = Word2Vec(sentences = [tokens], vector_size=100, window=5, min_count=1, workers=4) # 生成模型
    
    embeddings = []
    for i in range(len(tokens)):
        word_vectors = model.wv[tokens[i]]
        embeddings.append(word_vectors)
    
    weight_token_vector = []
    weights = list(token_weight_dict.values())
    for i in range(len(weights)):
        weight_token_vector.append((np.array(embeddings[i])*weights[i]).tolist())
    
    return weight_token_vector
