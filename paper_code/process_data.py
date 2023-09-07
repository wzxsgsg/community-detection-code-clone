import json
import os
import re
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, word2vec_embedding, final_centrality,compute_centrality,get_subgraph_centrality

nodetype = ['MethodDeclaration', 'Modifier', 'FormalParameter', 'ReferenceType', 'BasicType',
     'LocalVariableDeclaration', 'VariableDeclarator', 'MemberReference', 'ArraySelector', 'Literal',
     'BinaryOperation', 'TernaryExpression', 'StatementExpression','Assignment', 'MethodInvocation', 'Cast',
     'VariableDeclaration', 'ClassCreator','ArrayInitializer', 'Annotation','ArrayCreator', 'ConstructorDeclaration',
     'TypeArgument', 'EnhancedForControl', 'SuperMethodInvocation', 'SynchronizedStatement','InnerClassCreator', 'ExplicitConstructorInvocation',
     'ClassReference', 'SuperConstructorInvocation', 'ElementValuePair', 'AssertStatement',
     'ElementArrayValue', 'TypeParameter', 'FieldDeclaration', 'SuperMemberReference',
     'ContinueStatement', 'ClassDeclaration', 'TryResource', 'MethodReference',
     'LambdaExpression', 'InferredFormalParameter']

def save_data():
    path = '../input/codeclone_bcb/BCB/bigclonebenchdata'
#     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    count_lines = 1
    print(len(files))
    print("start save:")
    for file_name in files:
        if count_lines%500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name) # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        g_1, alltokens_1, x_1, g_edge_1 = create_ast_graph_func(path + '/' + file_name)  # 得到图、token、结点列表、边列表

        partition_1, val_dict_1, val_dict_num_1 = louvain_detect(g_1, x_1)

        subgraph_result_1 = get_subgraph_centrality(partition_1, g_1)

        subgraphs_1 = get_subgraph(partition_1, g_1)

        results_1, mutiply_centrality_1 = compute_centrality(subgraphs_1, subgraph_result_1)

        whole_graph_results_1 = get_whole_graph_centrality(g_1)

        final_centrality_result_1 = final_centrality(subgraphs_1, mutiply_centrality_1, whole_graph_results_1)

        token_weight_dict_1 = get_token_total_weight(x_1, final_centrality_result_1)

#         keys = list(token_weight_dict_1.keys())
#         for key in keys:
#             if key in nodetype:
#                 del token_weight_dict_1[key]


        with open('save_token_and_weight_8/save_token_weight_dict_'+result_1[0]+'.json', 'w') as f:
            json.dump(token_weight_dict_1, f)

    print("process end")

save_data()

import json
import os
import re
import pickle
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality
from gensim.models import Word2Vec


def save_token():
    path = 'save_token_and_weight_1'
    #     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(len(files))
    count_lines = 1
    print('load model:')
    model = Word2Vec.load("result/word2vec_model")
    print("start save:")
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name)  # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        try:
            with open(path + '/' + file_name, 'r') as f:
                data_1 = json.load(f)
        except Exception as e:
            print('Failed to open %s. Reason: %s' % (file_name, e))

        data_1 = dict(data_1)
        # print(data_1)
        weight_token_vector_1 = word2vec_embedding(model, data_1)
        with open('save_token_vector_1/weight_token_vector_' + result_1[0] + '.pkl', 'wb') as f:
            pickle.dump(weight_token_vector_1, f)

    print("process end")


save_token()

import json
import os
import re
import pickle
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality
from gensim.models import Word2Vec


def word2vec_train():
    path = 'save_token_and_weight_8'
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(len(files))
    count_lines = 1
    print("start train:")
    tokens = []
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name)  # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        try:
            with open(path + '/' + file_name, 'r') as f:
                data_1 = json.load(f)
        except Exception as e:
            print('Failed to open %s. Reason: %s' % (file_name, e))

        data_1 = dict(data_1)
        tokens.append(list(data_1.keys()))

    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)  # 生成模型
    model.save('result/word2vec_model')

    print("process end")


word2vec_train()

import json
import os
import re
import pickle
import numpy as np
from create_ast_graph import create_ast_graph_func
from gensim.models import Word2Vec


def save_original_ast_token():
    path = '../input/codeclone_bcb/BCB/bigclonebenchdata'
    #     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    count_lines = 1
    print(len(files))
    print('load model:')
    model = Word2Vec.load("result/word2vec_model")
    print("start save:")
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name)  # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        g_1, alltokens_1, x_1, g_edge_1 = create_ast_graph_func(path + '/' + file_name)  # 得到图、token、结点列表、边列表

        embeddings = []
        for i in range(len(x_1)):
            word_vectors = model.wv[x_1[i]]
            embeddings.append(np.array(word_vectors).tolist())

        with open('save_original_ast_token_vector/weight_token_vector_' + result_1[0] + '.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

    print("process end")


save_original_ast_token()

# 处理GCJ的数据
import json
import os
import re
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality, get_subgraph_centrality


def save_gcj_data():
    path = '../input/codeclone_gcj/googlejam4_src/12'
    #     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    count_lines = 1
    print(len(files))
    print("start save:")
    for file_name in files:
        if count_lines % 100 == 0:
            print("已处理文件：", count_lines)
        #         print(count_lines)
        count_lines += 1

        #         result_1 = re.findall(r'\d+\w', file_name) # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        g_1, alltokens_1, x_1, g_edge_1 = create_ast_graph_func(path + '/' + file_name)  # 得到图、token、结点列表、边列表

        partition_1, val_dict_1, val_dict_num_1 = louvain_detect(g_1, x_1)

        subgraph_result_1 = get_subgraph_centrality(partition_1, g_1)

        subgraphs_1 = get_subgraph(partition_1, g_1)

        results_1, mutiply_centrality_1 = compute_centrality(subgraphs_1, subgraph_result_1)

        whole_graph_results_1 = get_whole_graph_centrality(g_1)

        final_centrality_result_1 = final_centrality(subgraphs_1, mutiply_centrality_1, whole_graph_results_1)

        token_weight_dict_1 = get_token_total_weight(x_1, final_centrality_result_1)

        #         keys = list(token_weight_dict_1.keys())
        #         for key in keys:
        #             if key in nodetype:
        #                 del token_weight_dict_1[key]

        with open('GCJ_data/Json6/' + file_name + '.json', 'w') as f:
            json.dump(token_weight_dict_1, f)

    print("process end")


save_gcj_data()

import json
import os
import re
import pickle
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality
from gensim.models import Word2Vec


def word2vec_GCJ_train():
    path = 'GCJ_data/Json6'
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(len(files))
    count_lines = 1
    print("start train:")
    tokens = []
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        #         result_1 = re.findall(r'\d+\w', file_name) # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        try:
            with open(path + '/' + file_name, 'r') as f:
                data_1 = json.load(f)
        except Exception as e:
            print('Failed to open %s. Reason: %s' % (file_name, e))

        data_1 = dict(data_1)
        tokens.append(list(data_1.keys()))

    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)  # 生成模型
    model.save('result/word2vec_GCJ_model')

    print("process end")


word2vec_GCJ_train()

import json
import os
import re
import pickle
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality
from gensim.models import Word2Vec


def save_GCJ_token():
    path = 'GCJ_data/Json6'
    #     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(len(files))
    count_lines = 1
    print('load model:')
    model = Word2Vec.load("result/word2vec_GCJ_model")
    print("start save:")
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        #         result_1 = re.findall(r'\d+\w', file_name) # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        try:
            with open(path + '/' + file_name, 'r') as f:
                data_1 = json.load(f)
        except Exception as e:
            print('Failed to open %s. Reason: %s' % (file_name, e))

        data_1 = dict(data_1)
        # print(data_1)
        weight_token_vector_1 = word2vec_embedding(model, data_1)
        with open('GCJ_data/Token6/' + file_name[0: -5] + '.pkl', 'wb') as f:
            pickle.dump(weight_token_vector_1, f)

    print("process end")


save_GCJ_token()

import json
import os
import re
from create_ast_graph import create_ast_graph_func
from la_PCA import main


def save_data():
    path = '../input/codeclone_bcb/BCB/bigclonebenchdata'
    #     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    count_lines = 1
    print(len(files))
    print("start save:")
    for file_name in files:
        #         print(count_lines)
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name)  # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        token_weight_dict = main(path + '/' + file_name)

        with open('save_la_PCA_token_weight_dict_1/save_token_weight_dict_' + result_1[0] + '.json', 'w') as f:
            json.dump(token_weight_dict, f)

    print("process end")


save_data()

import json
import os
import re
import pickle
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality
from gensim.models import Word2Vec


def word2vec_train():
    path = 'save_la_PCA_token_weight_dict'
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(len(files))
    count_lines = 1
    print("start train:")
    tokens = []
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name)  # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        try:
            with open(path + '/' + file_name, 'r') as f:
                data_1 = json.load(f)
        except Exception as e:
            print('Failed to open %s. Reason: %s' % (file_name, e))

        data_1 = dict(data_1)
        tokens.append(list(data_1.keys()))

    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)  # 生成模型
    model.save('result/la_PCA_word2vec_model')

    print("process end")


word2vec_train()

import json
import os
import re
import pickle
from create_ast_graph import create_ast_graph_func
from create_louvain import louvain_detect, get_subgraph, get_whole_graph_centrality, get_token_total_weight, \
    word2vec_embedding, final_centrality, compute_centrality
from gensim.models import Word2Vec


def save_token():
    path = 'save_la_PCA_token_weight_dict_1'
    #     files = os.listdir(path)
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    print(len(files))
    count_lines = 1
    print('load model:')
    model = Word2Vec.load("result/la_PCA_word2vec_model")
    print("start save:")
    for file_name in files:
        if count_lines % 500 == 0:
            print("已处理文件：", count_lines)
        count_lines += 1
        result_1 = re.findall(r'\d+\w', file_name)  # 得到相应的文件名，比如17874921.txt，得到17874921这一串数字

        try:
            with open(path + '/' + file_name, 'r') as f:
                data_1 = json.load(f)
        except Exception as e:
            print('Failed to open %s. Reason: %s' % (file_name, e))

        data_1 = dict(data_1)
        # print(data_1)
        weight_token_vector_1 = word2vec_embedding(model, data_1)
        with open('save_la_PCA_token_vector_1/weight_token_vector_' + result_1[0] + '.pkl', 'wb') as f:
            pickle.dump(weight_token_vector_1, f)

    print("process end")


save_token()
