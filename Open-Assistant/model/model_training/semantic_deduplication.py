# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import json
from datasets import load_dataset, Dataset
from torch.utils.data import ConcatDataset
import torch
from tqdm import tqdm

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from math import ceil

from sentence_transformers import SentenceTransformer, util

from custom_datasets.instruction import InstructionDataset
from custom_datasets.MyDialogue import MyDialogue

from enum import Enum


class DatasetType(Enum):
    SingleTurn = 1
    MultiTurn = 2


if __name__ == '__main__':
    # 数据读取-支持单轮和多轮格式
    # 先从oa中读取dataset的config，将这些数据集concat起来，先生成embedding，然后使用多进程计算相似度去重。
    # 使用bit_map表示数据点是否保留。
    # 最后根据bit_map分别保存留下的数据，保存分为两种格式：单轮对话，多轮对话

    # configs
    threshold = 0.88

    from trainer_sft import argument_parsing
    training_conf = argument_parsing()
    from utils import get_dataset, get_dataset_name_and_kwargs_from_data_config, get_one_dataset
    from torch.utils.data import ConcatDataset, Dataset, Subset


    def get_dataset(
            conf,
            mode: str = "sft",
    ) -> tuple[ConcatDataset, dict[str, Subset]]:
        train_datasets = []
        dataset_names = []
        print("Creating datasets...")
        for data_config in tqdm(conf.datasets + conf.datasets_extra):
            dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
            print(f"getting dataset: {dataset_name}")
            train, val = get_one_dataset(conf, dataset_name, mode=mode, **kwargs)
            train_datasets.append(train)

            dataset_names.append(dataset_name)
        train = ConcatDataset(train_datasets)

        return train, dataset_names

    train_datasets, dataset_names = get_dataset(training_conf)
    print(dataset_names)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    train_datasets_len = len(train_datasets)

    print(''.join(train_datasets[0]))

    data_idx_list = [i for i in range(len(train_datasets))]
    bit_map = np.array([True] * len(train_datasets))
    bit_map_f = np.memmap('./bitmap_cache', dtype='bool', mode='w+', shape=(len(train_datasets)))
    bit_map_f[:] = bit_map[:]

    emb_cache = np.memmap('./emb_cache', dtype='float32', mode='w+', shape=(len(train_datasets), 384))

    
    # emb_befor_clean = []

    # pre_cache
    from tqdm import trange
    for idx in trange(len(train_datasets)):
        cur_data = ''.join(train_datasets[idx])
        emb = model.encode(cur_data)
        # emb_befor_clean.append(emb)
        emb_cache[idx,:] = emb

    
    # # plot before clean
    # temp_highdim = np.array(emb_befor_clean)
    # x_tsne = TSNE(n_components=2, learning_rate=100, random_state=501).fit_transform(temp_highdim)

    # plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=20)
    # plt.legend(fontsize=10)
    # plt.title("before clean")
    # plt.show()

    def calc_similarity(bit_map, data_idx_chunk):

        global emb_cache

        for idx_i in tqdm(data_idx_chunk):
            if not bit_map[idx_i]:
                continue
            cur_data = ''.join(train_datasets[idx_i])
            for idx_j in data_idx_list:
                if not bit_map[idx_j]:
                    continue
                if idx_i == idx_j:
                    continue
                cmp_data = ''.join(train_datasets[idx_j])

                emb1 = emb_cache[idx_i,:]

                emb2 = emb_cache[idx_j,:]

                cos_sim = util.cos_sim(emb1, emb2)
                cos_sim = cos_sim.item()

                if cos_sim > threshold:
                    # now: keep the longer one
                    # todo: keep the one with lower ppl
                    with open('sim1.txt', 'a+') as f:
                        f.write(cur_data + '\n')
                        f.write(cmp_data + '\n')
                        f.write(str(cos_sim) + '\n\n')


                    if len(cur_data) > len(cmp_data):
                        bit_map[idx_j] = False
                    else:
                        bit_map[idx_i] = False
                        break

    # 数据去重


    num_cpus = cpu_count()
    docs = data_idx_list
    chunk_size = ceil(len(docs) / num_cpus)
    chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
    # 将数据切成cpu个数份
    Parallel(n_jobs=num_cpus)(delayed(calc_similarity)(bit_map=bit_map_f, data_idx_chunk=chunk) for chunk in chunks)

    torch.save(list(bit_map_f), 'bit_map.pt') # save in case error

    def get_dataset_type(data_set):

        if isinstance(data_set, InstructionDataset):
            return DatasetType.SingleTurn

        if isinstance(data_set, MyDialogue):
            return DatasetType.MultiTurn

        for d in data_set:
            if len(d) > 2:
                return DatasetType.MultiTurn

        return DatasetType.SingleTurn

    # 数据保存

    print("Saving Results...")

    global_idx = -1

    import os
    save_path_root = './cleaned_data/'
    os.makedirs(save_path_root, exist_ok=True)

    for dataset_idx, train_dataset in enumerate(dataset_names):

        dataset_type = get_dataset_type(train_dataset)
        cur_path = path_list[dataset_idx]
        cur_path = cur_path.split('/')[-1].split('.')[0]

        with open(os.path.join(save_path_root, cur_path+'.jsonl'), 'w+') as f:
            if dataset_type == DatasetType.SingleTurn:
                for idx in range(len(train_dataset)):
                    global_idx += 1
                    if not bit_map[global_idx]:
                        continue

                    cur_data = train_dataset[idx]
                    cur_data_dict = {"INSTRUCTION":cur_data[0], "RESPONSE":cur_data[1], "SOURCE": cur_path, "METADATA": None}
                    f.write(json.dumps(cur_data_dict, ensure_ascii=False) + '\n')

            elif dataset_type == DatasetType.MultiTurn:
                role_dict = {
                    0:"prompter",
                    1:"assistant"
                }

                for idx in range(len(train_dataset)):
                    global_idx += 1
                    if not bit_map[global_idx]:
                        continue

                    cur_data = train_dataset[idx]
                    cur_role = 1
                    pre_data_dict = None
                    cur_data_dict = None
                    for sub_turn in cur_data[::-1]:
                        if cur_data_dict is None:
                            cur_data_dict = {"text":sub_turn, "role":role_dict[cur_role], "meta":{}, "replies":[]}
                        else:
                            pre_data_dict = cur_data_dict
                            cur_data_dict = {"text":sub_turn, "role":role_dict[cur_role], "meta":{}, "replies":[pre_data_dict]}

                        if cur_role == 0:
                            cur_role = 1
                        elif cur_role == 1:
                            cur_role = 0
                    cur_data = {"thread": cur_data_dict, "source":cur_path, "meta":{}}
                    f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

    # # plot after clean
    # emb_after_clean = []
    # for idx in range(len(train_datasets)):
    #     if not bit_map_f[idx]:
    #         continue
    #     cur_data = ''.join(train_datasets[idx])
    #     emb = emb_cache[idx,:]
    #     emb_after_clean.append(emb)

    # temp_highdim = np.array(emb_after_clean)

    # x_tsne = TSNE(n_components=2, learning_rate=100, random_state=501).fit_transform(temp_highdim)
    # plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=20)
    # plt.legend(fontsize=10)
    # plt.title("after clean")
    # plt.show()

    # show clean statistics
    print(f"before clean len: {len(train_datasets)}")
    print(f"after clean len: {list(bit_map_f).count(True)}")





