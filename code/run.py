import os
import time
import json
import argparse
from argparse import Namespace
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import LRGCPND
from utils import load_data, load_samples
from metrics import evaluate


os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_num', type=int, default=625, help='Number of ncRNAs.')
    parser.add_argument('--d_num', type=int, default=121, help='Number of drugs.')
    parser.add_argument('-K', type=int, default=4, help='Depth of layers.')
    parser.add_argument('-S', type=int, default=32, help='Embedding size.')
    parser.add_argument('-r', '--reg', type=float, default=0.05, help='Coefficient of L2 regularization.')
    parser.add_argument('-l', '--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size to train.')
    parser.add_argument('-f', '--fold', type=int, default=5, help='Number of folds for cross validation.')
    parser.add_argument('-t', '--time', type=str, default=None, help='Timestamp in milliseconds for training.')
    parser.add_argument('--save_models', action='store_true', default=False, help='Save trained models.')
    parser.add_argument('--have_trained', action='store_true', default=False, help='Have trained models.')
    args = parser.parse_args()
    return args


def train(data_path, n_num, d_num, K, E_size, reg, lr, epochs, batch_size):
    samples, adj = load_data(data_path, n_num, d_num)
    train_dataset = TensorDataset(torch.LongTensor(samples))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)

    model = LRGCPND(n_num, d_num, adj, K, E_size, reg)
    model=model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        model.train()
        loss_sum = []
        start_time = time.time()

        for data in train_loader:
            data = data[0].cuda()
            n = data[:, 0]
            d_i = data[:, 1]
            d_j = data[:, 2]

            optimizer.zero_grad()
            pre_i, pre_j, loss = model(n, d_i, d_j)
            loss.backward()
            optimizer.step()

            loss_sum.append(loss.item())

        train_loss=round(np.mean(loss_sum),4)
        end_time = time.time()
        str_e = 'epoch:%-3d\ttime:%.2f\ttrain loss:%.4f' % (e, end_time-start_time, train_loss)
        print(str_e)
    return model


def test(model, data_path, batch_size, threshold):
    model.eval()
    samples = load_samples(data_path)
    test_dataset = TensorDataset(torch.LongTensor(samples))
    test_loader = DataLoader(test_dataset, batch_size)
    results, _ = evaluate(model, test_loader, threshold)
    return results


def run(args):
    data_path_base = '../data/samples/'
    model_path_base = '../model'
    result_path_base = '../result'
    str_time = str(round(time.time()*1000))

    if args.save_models:
        model_path_base += '/' + str_time
        os.makedirs(model_path_base)
    if not (os.path.exists(result_path_base)):
        os.makedirs(result_path_base)

    avg_res = defaultdict(int)
    print('-'*20 + 'start' + '-'*20)
    print('time: ' + str_time)
    print(vars(args))
    for i in range(args.fold):
        train_path = data_path_base + 'train_' + str(i) + '.txt'
        test_path = data_path_base + 'test_' + str(i) + '.txt'
        print('-'*20 + ('fold-%d-start' % i) + '-'*20)
        model = train(train_path, args.n_num, args.d_num, args.K, args.S, args.reg, args.lr, args.epochs, args.batch)

        if args.save_models:
            torch.save(model.state_dict(), model_path_base + '/' + str(i) + '.pt')
        
        print('-'*20 + ('fold-%d-end' % i) + '-'*20)
        results = test(model, test_path, 32, 0)
        for k, v in results.items():
            print('%s\t%.4f' % (k, v))
            avg_res[k] += v
    
    print('-'*20 + 'end' + '-'*20)
    print('time: ' + str_time)
    print_average_results(args, avg_res)

    res = dict()
    res['args'] = str(args)
    # res['args'] = vars(args)
    res['performance'] = avg_res
    with open(result_path_base + '/' + str_time + '.json', 'w') as f:
        f.write(json.dumps(res))


def load_args(str_time):
    assert str_time, 'illegal timestamp'

    result_path = '../result/' + str_time + '.json'
    with open(result_path, 'r') as f:
        res = json.loads(f.read())
        args = eval(res['args'])
        # args = res['args']
    return args


def load_and_test(str_time):
    data_path_base = '../data/samples/'
    model_path_base = '../model'

    args = load_args(str_time)
    args.have_trained = True
    args.time = str_time
    avg_res = defaultdict(int)

    for i in range(args.fold):
        train_path = data_path_base + 'train_' + str(i) + '.txt'
        test_path = data_path_base + 'test_' + str(i) + '.txt'

        _, adj = load_data(train_path, args.n_num, args.d_num)
        model = LRGCPND(args.n_num, args.d_num, adj, args.K, args.S, args.reg)
        model=model.to('cuda')

        model_path = model_path_base + '/' + str_time + '/' + str(i) + '.pt'
        model.load_state_dict(torch.load(model_path))

        print('-'*20 + ('fold-%d' % i) + '-'*20)
        results = test(model, test_path, 32, 0)
        for k, v in results.items():
            print('%s\t%.4f' % (k, v))
            avg_res[k] += v
    
    print_average_results(args, avg_res)


def print_average_results(args, avg_res):
    print('-'*20 + 'Args' + '-'*20)
    print(vars(args))
    for k, v in avg_res.items():
        avg_res[k] = round(v / 5, 4)
    print('-'*20 + 'AVG' + '-'*20)
    print(avg_res)


if __name__ == '__main__':
    args = get_args()
    if args.have_trained:
        load_and_test(args.time)
    else:
        run(args)
