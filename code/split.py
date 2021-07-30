import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.utils import resample


def k_fold_split(index_path, matrix_path, k, ratio=1):
    index_df = pd.read_table(index_path, header=None)
    split_df = pd.read_csv(matrix_path, index_col=0)
    split_arr = split_df.values

    row_num, col_num = split_arr.shape
    pos_lists = []
    neg_lists = []
    count = 0
    for i in range(row_num):
        rna_vt = split_arr[i]
        pos_index_arr = np.where(rna_vt==1)[0]
        neg_index_arr = np.where(rna_vt==0)[0]
        assert len(pos_index_arr) + len(neg_index_arr) == col_num, 'length doesn\'t match while loading data'
        pos_lists.append(pos_index_arr)
        neg_lists.append(neg_index_arr)
        count+=len(pos_index_arr)
    print(count)


    k_fold = []
    index = set(range(index_df.shape[0]))
    for i in range(k):
        tmp = random.sample(list(index), int(1.0 / k * index_df.shape[0]))
        k_fold.append(tmp)
        index -= set(tmp)
    

    if len(index)!= 0:
        picked = np.arange(k)
        np.random.shuffle(picked)
        picked = picked[:len(index)]
        for n, i in enumerate(index):
            k_fold[picked[n]].append(i)

    
    data_path_base = '../data/samples'
    if not (os.path.exists(data_path_base)):
        os.makedirs(data_path_base)
    for i in range(k):
        print("Fold-{}........".format(i + 1))
        tra = []
        dev = k_fold[i]
        for j in range(k):
            if i != j:
                tra += k_fold[j]
        train_pos = index_df.iloc[tra].values
        test_pos = index_df.iloc[dev].values

        train_samples, test_samples = sample(train_pos, test_pos, neg_lists, ratio)
        write_samples(train_samples, data_path_base + '/train_'+str(i)+'.txt')
        write_samples(test_samples, data_path_base + '/test_'+str(i)+'.txt')

    print("done!")


def sample(train_pos, test_pos, neg_lists, ratio):
    train_cast_dict = cast_to_dict(train_pos)
    test_cast_dict = cast_to_dict(test_pos)
    train_samples = list()
    test_samples = list()

    for k in (set(train_cast_dict.keys()) | set(test_cast_dict.keys())):
        neg_list = neg_lists[k]
        random.shuffle(neg_list)

        bef_train_len = len(train_samples)
        bef_test_len = len(test_samples)

        neg_list, train_pick_samples = check_and_add(k, train_cast_dict, neg_list, train_samples, ratio)
        neg_list, test_pick_samples = check_and_add(k, test_cast_dict, neg_list, test_samples, ratio)

        aft_train_len = len(train_samples)
        aft_test_len = len(test_samples)
        assert aft_train_len == bef_train_len + len(train_pick_samples), 'illegal triples while generating data'
        assert aft_test_len == bef_test_len + len(test_pick_samples), 'illegal triples while generating data'


        if len(train_pick_samples) and len(test_pick_samples):
            if len(set(np.array(train_pick_samples)[:,2]) & set(np.array(test_pick_samples)[:,2])) != 0:
                print(train_pick_samples, test_pick_samples, sep='\n')
                raise Exception('illegal triples while generating data')
    return train_samples, test_samples


def cast_to_dict(pos_samples):
    cast_dict = defaultdict(set)
    for s in pos_samples:
        assert len(s) == 3, 'length doesn\'t match while loading data'
        cast_dict[int(s[0])].add(int(s[1]))
    return cast_dict


def check_and_add(k, cast_dict, neg_list, samples, ratio):
    pick_samples = list()
    if k in cast_dict.keys():
        v = cast_dict[k]
        sample_neg_list = resample(neg_list, n_samples=ratio*len(v), replace=False)
        neg_list = list(np.setdiff1d(np.array(neg_list), np.array(sample_neg_list), assume_unique=True))
        
        for i, d_pos in enumerate(v):
            d_negs = sample_neg_list[i*ratio:(i+1)*ratio]
            for d_neg in d_negs:
                pick_samples.append([k, d_pos, d_neg])            
        samples.extend(pick_samples)
    return neg_list, pick_samples


def write_samples(samples, path):
    with open(path, 'w') as f:
        for i in samples:
            assert len(i) == 3, 'length doesn\'t match while writing samples'
            f.write(str(i[0])+'\t'+str(i[1])+'\t'+str(i[2])+'\n')


if __name__ == '__main__':
    relation_index_path =  '../data/relation_index_split.txt'
    relation_matrix_path = '../data/ncrna-drug_split.csv'
    k_fold_split(relation_index_path, relation_matrix_path, 5)
