import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm.notebook import tqdm, trange
from time import sleep
from create_data import create_data
from create_GCJ_data import create_gcj_data
from word2vec import LSTM, SiameseGRU
from attention_LSTM import SiameseLSTM
from transformer import TransformerModel
import time
import psutil
import os

info = psutil.virtual_memory()
# 总内存
print(u'总内存：', info.total)

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--data_setting", default='11')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=1)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0.5)
args = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

train_data, valid_data, test_data = create_data(args.data_setting)
# train_data, valid_data, test_data = create_gcj_data(args.data_setting)

# model = LSTM(100, 256, 50).to(device)
model = SiameseLSTM(100, 256, 50).to(device)
# model = TransformerModel(100, 50, d_model=100, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128, dropout=0.1).to(device)
# model = TransformerModel(100, 50)
# model = SiameseGRU(100, 256, args.num_layers, 50).to(device)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
criterion1 = nn.CrossEntropyLoss()


def create_batches(data):
    # random.shuffle(data)
    batches = [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]
    return batches


def test(dataset):
    # model.eval()
    count = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = []
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        input_1, input_2 = data
        input_1 = torch.tensor(input_1, device=device)
        input_2 = torch.tensor(input_2, device=device)
        output_1, output_2 = model(input_1, input_2)
        output = F.cosine_similarity(output_1, output_2, dim=1, eps=1e-8)
        results.append(output.item())
        #         prediction = torch.sign(output).item()
        prediction = output.item()

        if prediction > args.threshold and label.item() == 1:
            tp += 1
            # print('tp')
        if prediction <= args.threshold and label.item() == -1:
            tn += 1
            # print('tn')
        if prediction > args.threshold and label.item() == -1:
            fp += 1
            # print('fp')
        if prediction <= args.threshold and label.item() == 1:
            fn += 1
            # print('fn')
    print(tp, tn, fp, fn)
    p = 0.0
    r = 0.0
    f1 = 0.0
    if tp + fp == 0:
        print('precision is none')
        return
    p = tp / (tp + fp)
    if tp + fn == 0:
        print('recall is none')
        return
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    return results, p, r, f1, tp, tn, fp, fn


epochs = trange(args.num_epochs, leave=True, desc="Epoch")
best_f1 = None
for epoch in epochs:  # without batching
    batches = create_batches(train_data)
    totalloss = 0.0
    main_index = 0.0
    train_time_start = time.time()
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        #         sleep(0.01)
        optimizer.zero_grad()
        batchloss = 0
        for data, label in batch:
            label = torch.tensor(label, dtype=torch.float, device=device)
            #             label = torch.tensor([label], dtype=torch.long, device=device)
            input_1, input_2 = data
            input_1 = torch.tensor(input_1, device=device)
            input_2 = torch.tensor(input_2, device=device)
            #             print(input_1.shape)
            #             print(input_2.shape)
            output_1, output_2 = model(input_1, input_2)
            #             print(output_1.shape)
            #             print(output_2.shape)
            output = F.cosine_similarity(output_1, output_2, dim=1, eps=1e-8)
            #             cossim = cossim.unsqueeze(0)
            #             print(cossim.shape)
            output = output.squeeze(0)
            batchloss = batchloss + criterion(output, label)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
    #     print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
    #     print(u'内存占比：',info.percent)
    memory_used = psutil.Process(os.getpid()).memory_info().rss
    memory_precent = info.percent

    train_time_end = time.time()
    train_time = train_time_end - train_time_start

    valid_time_start = time.time()
    devresults, valid_p, valid_r, valid_f1, valid_tp, valid_tn, valid_fp, valid_fn = test(valid_data)
    valid_time_end = time.time()
    valid_time = valid_time_end - valid_time_start
    devfile = open('result/6/dev_epoch_' + str(epoch + 1), mode='w')
    for res in devresults:
        devfile.write(str(res) + '\n')
    devfile.close()
    if best_f1 is None or best_f1 < valid_f1:
        best_f1 = valid_f1
        torch.save(model.state_dict(), 'result/best_model')
    #         torch.save(model.state_dict(), 'result/best_GCJ_model')
    #         torch.save(model.state_dict(), 'result/best_PCA_model')

    #     model.load_state_dict(torch.load('result/best_model'))
    #     model.load_state_dict(torch.load('result/best_GCJ_model'))

    print('start test:')
    test_time_start = time.time()
    testresults, test_p, test_r, test_f1, test_tp, test_tn, test_fp, test_fn = test(test_data)
    test_time_end = time.time()
    test_time = test_time_end - test_time_start
    #     resfile = open('result/6/test_result_epoch_' + str(epoch + 1), mode='w')
    #     for res in testresults:
    #         resfile.write(str(res) + '\n')
    #     resfile.close()
    finalfile = open('result/6/test_final_result_epoch_' + str(epoch + 1), mode='w')
    finalfile.write('threshold: ' + str(args.threshold) + '\n')
    finalfile.write('train_time:' + str(train_time) + '\n')
    finalfile.write('valid_time:' + str(valid_time) + '\n')
    finalfile.write('test_time:' + str(test_time) + '\n')
    finalfile.write('内存使用：' + str(memory_used) + '\n')
    finalfile.write('内存占比：' + str(memory_precent) + '\n')
    finalfile.write(str(test_tp) + ' ')
    finalfile.write(str(test_tn) + ' ')
    finalfile.write(str(test_fp) + ' ')
    finalfile.write(str(test_fn) + '\n')
    finalfile.write('loss:' + str(totalloss / main_index) + '\n')
    finalfile.write(str(test_p) + '\n')
    finalfile.write(str(test_r) + '\n')
    finalfile.write(str(test_f1) + '\n')
    finalfile.close()