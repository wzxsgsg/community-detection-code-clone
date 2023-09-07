from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, word2vec_embedding, final_centrality,compute_centrality
import re
import numpy as np
import json
import pickle
import os

def create_data(id):
    indexdir = "../input/codeclone_bcb/BCB/"
    if id == '0':
        train_file = open(indexdir + 'traindata.txt')
        valid_file = open(indexdir + 'devdata.txt')
        test_file = open(indexdir + 'testdata.txt')
    elif id == '11':
        train_file = open(indexdir + 'traindata11.txt')
        valid_file = open(indexdir + 'devdata.txt')
        test_file = open(indexdir + 'testdata.txt')
    else:
        print('file not exist')
        quit()
    train_list = train_file.readlines()
    valid_list = valid_file.readlines()
    test_list = test_file.readlines()

    print('train data')
    train_data = create_pair_data(train_list)
    print('valid data')
    valid_data = create_pair_data(valid_list)
    print('test data')
    test_data = create_pair_data(test_list)
    return train_data, valid_data, test_data


def create_pair_data(path_list):
    data_list = []
    count_lines = 1
    positive = 0
    nagative = 0
    print("start process:")
    
    # 读取pkl文件
    data_dict = {}
    pkl_folder = "save_token_vector_6"  # pkl文件夹路径
#     pkl_folder = 'save_original_ast_token_vector'
    for filename in os.listdir(pkl_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(pkl_folder, filename), "rb") as f:
                data = pickle.load(f)
            data_dict[filename] = data
            
    print("得到所有文件的对应内容")
            
    for line in path_list:
#         if count_lines%100 == 0:
#             print(count_lines)
        count_lines += 1
        pair_info = line.split()
        code_1_path = "../input/codeclone_bcb/BCB" + pair_info[0].strip('.')
        code_2_path = "../input/codeclone_bcb/BCB" + pair_info[1].strip('.')
        label = int(pair_info[2])
        
        if label == 1:
            positive += 1
        else:
            nagative += 1
        
        result_1 = re.findall(r'\d+\w', code_1_path)
        result_2 = re.findall(r'\d+\w', code_2_path)
        
        # 读取pkl文件
        weight_token_vector_1 = data_dict['weight_token_vector_'+ result_1[0] +'.pkl']
        weight_token_vector_2 = data_dict['weight_token_vector_'+ result_2[0] +'.pkl']

        data_1 = [weight_token_vector_1]
        data_2 = [weight_token_vector_2]

        data = [[data_1, data_2], label]
        data_list.append(data)
    
    print('positive:', positive)
    print('nagative:', nagative)
    print()
    return data_list