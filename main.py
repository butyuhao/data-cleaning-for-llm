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

from enum import Enum


class DatasetType(Enum):
    SingleTurn = 1
    MultiTurn = 2

MY_INSTRUCTION_DATASETS = {
"old_chatglm_alpaca": "/data/ecnu/data/old_Chatglm_data/old_chatglm_alpaca.json",
"old_chatglm_belle": "/data/ecnu/data/old_Chatglm_data/old_chatglm_Belle.json",
"old_chatglm_belle_math":"/data/ecnu/data/old_Chatglm_data/old_chatglm_BELLE_math.json",
"_alpaca": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/alpaca.jsonl",
"_belle": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/Belle.jsonl",
"_belle_generated_chat": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/BELLE_generated_chat.jsonl",
"_belle_math": "./data/opensource_data/BELLE_math.jsonl",
"_belle_train_2m": "./data/opensource_data/BELLE_train_2M.jsonl",
"_coig_single": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/COIG_single.jsonl",
"_firefly": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/firefly.jsonl",
"_flan_cot": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/Flan_CoT.jsonl",
"_instinwild_ch": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/instinwild_ch.jsonl",
"_instinwild_en": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/instinwild_en.jsonl",
"_unnatural_instruction": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/opensource_data/Unnatural_Instruction.jsonl",
"_composition_review": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/composition_review.jsonl",
"_correct": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/correct.jsonl",
"_poem_generate": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/poem_generate.jsonl",
"_poem_transform": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/poem_transform.jsonl",
"_reading_comprehension_en": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/reading_comprehension_en.jsonl",
"_rewriting": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/rewriting.jsonl",
"_story_generate": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/story_generate.jsonl",
"_summary": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/summary.jsonl",
"_transstyle": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/TransStyle.jsonl",
"_writing": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/writing.jsonl",
"_writing_cot": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/writing_cot.jsonl",
"_chengyu_gushi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/chengyu/chengyu_gushi.jsonl",
"_chengyu_jieshi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/chengyu/chengyu_jieshi.jsonl",
"_chengyu_laiyuan": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/chengyu/chengyu_laiyuan.jsonl",
"_chengyu_liju": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/chengyu/chengyu_liju.jsonl",
"_chengyu_pinyin": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/chengyu/chengyu_pinyin.jsonl",
"_jieshi_chengyu": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/chengyu/jieshi_chengyu.jsonl",
"_biaoti_chaodai": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/biaoti_chaodai.jsonl",
"_biaoti_gushi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/biaoti_gushi.jsonl",
"_biaoti_leixing": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/biaoti_leixing.jsonl",
"_biaoti_mingju": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/biaoti_mingju.jsonl",
"_biaoti_zuozhe": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/biaoti_zuozhe.jsonl",
"_gushi-beijing_yiwen": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi-beijing_yiwen.jsonl",
"_gushi_beijing": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_beijing.jsonl",
"_gushi_biaoti": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_biaoti.jsonl",
"_gushi_chaodai": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_chaodai.jsonl",
"_gushi_gushi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_gushi.jsonl",
"_gushi_leixing": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_leixing.jsonl",
"_gushi_mingju": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_mingju.jsonl",
"_gushi_shangxi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_shangxi.jsonl",
"_gushi_yiwen": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_yiwen.jsonl",
"_gushi_zuozhe": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/gushi_zuozhe.jsonl",
"_leixing_biaoti": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/leixing_biaoti.jsonl",
"_leixing_gushi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/leixing_gushi.jsonl",
"_leixing_zuozhe": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/leixing_zuozhe.jsonl",
"_mingju_biaoti-yuanwen": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/mingju_biaoti-yuanwen.jsonl",
"_mingju_zuozhe": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/mingju_zuozhe.jsonl",
"_shiren_chaodai": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/shiren_chaodai.jsonl",
"_shiren_jieshao": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/shiren_jieshao.jsonl",
"_zuozhe-leixing_biaoti": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/zuozhe-leixing_biaoti.jsonl",
"_zuozhe_gushi": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/zuozhe_gushi.jsonl",
"_zuozhe_leixing": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/zuozhe_leixing.jsonl",
"_zuozhe_mingju": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/gushi/zuozhe_mingju.jsonl",
"_daan_miyu": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/miyu/daan_miyu.jsonl",
"_miyu_daan": "/nas-alinlp/butyuhao/data/educhat_data/datasets/lab_data_normal_quality/zeroshot_data/miyu/miyu_daan.jsonl",
}


INSTRUCTION_DATASETS = {
    # Note humaneval_mbpp_codegen_qa returns a code string that we would want to at least wrap in ``` marks`
    "humaneval_mbpp_codegen_qa": "OllieStanley/humaneval-mbpp-codegen-qa",
    # Write unit tests to do task X
    "humaneval_mbpp_testgen_qa": "OllieStanley/humaneval-mbpp-testgen-qa",
    "grade_school_math_instructions": "qwedsacf/grade-school-math-instructions",
    "recipes": "dctanner/oa_recipes",
    "ubuntu_dialogue_qa": "sedthh/ubuntu_dialogue_qa",
    "cmu_wiki_qa": "sedthh/cmu_wiki_qa",
    "youtube_subs_howto100m": "totuta/youtube_subs_howto100M",
    "iapp_wiki_qa_squad": "wannaphong/iapp_wiki_qa_squad_oa",
    "zhihu-kol": "wangrui6/zhihu-kol",
    "minimath": "kentsui/minimath",
    "oa_wiki_qa_bart_10000row": "michaelthwan/oa_wiki_qa_bart_10000row",
    "oa_leet10k": "ehartford/oa_leet10k",
    "poem_instructions": "checkai/instruction-poems",

}

class MyDialogue(Dataset):
    def __init__(self, mode: str, cache_dir: str = None) -> None:
        self.mode = mode
        self.rows = []
        with open(cache_dir,"r",encoding="utf-8") as f:
            self.rows = f.readlines()
        def get(x):
            l = []
            x = x["thread"]
            while True:
                l.append(x["text"])
                if len(x["replies"])==0:
                    break
                x = x["replies"][0]
            return l
        for i in range(len(self.rows)):
            self.rows[i] = get(json.loads(self.rows[i]))
            # print(self.rows[i])


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        dialogue: list = self.rows[index]
        if self.mode == "sft":
            return dialogue
        elif self.mode == "rl":
            return tuple(dialogue[:-1])

class InstructionDataset(Dataset):
    def __init__(self, dataset, cache_dir, split, max_words=512):
        self.name = dataset
        print(dataset,cache_dir)
        if dataset in MY_INSTRUCTION_DATASETS:
            self.dataset = load_dataset("json",data_files=MY_INSTRUCTION_DATASETS[dataset], split=split)
        else:
            self.dataset = load_dataset(INSTRUCTION_DATASETS[dataset], cache_dir=cache_dir, split=split)
        self.instruction_column = "INSTRUCTION" if dataset != "minimath" else "question"
        self.response_column = "RESPONSE" if dataset != "minimath" else "answer"
        self.max_words = max_words

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return (data[self.instruction_column], data[self.response_column])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 数据读取-支持单轮和多轮格式
    # 去重复的时候认为是一个大数据集，保存的时候再根据bit_map分别进行保存
    threshold = 0.88

    belle_dialog_path = './data/opensource_data/BELLE_dialog.jsonl'
    belle_dialog_dataset = MyDialogue(mode='sft', cache_dir=belle_dialog_path)
    belle_math_dataset = InstructionDataset('_belle_math', cache_dir='./cache', split='train')
    belle_train_2m_dataset = InstructionDataset('_belle_train_2m', cache_dir='./cache', split='train')
    path_list = ['./data/opensource_data/BELLE_dialog.jsonl', './data/opensource_data/BELLE_math.jsonl', './data/opensource_data/BELLE_train_2M.jsonl']

    train_datasets_list = [belle_dialog_dataset, belle_math_dataset, belle_train_2m_dataset]

    train_datasets = ConcatDataset(train_datasets_list)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_datasets_len = len(train_datasets)

    print(''.join(train_datasets[0]))

    data_idx_list = [i for i in range(len(train_datasets))]
    bit_map = np.array([True] * len(train_datasets))
    bit_map_f = np.memmap('./bitmap_cache', dtype='bool', mode='w+', shape=(len(train_datasets)))
    bit_map_f[:] = bit_map[:]

    emb_cache = np.memmap('./emb_cache', dtype='float32', mode='w+', shape=(len(train_datasets), 384))

    from lru import LRU
    # lru_cache = LRU(2000) # 内存不够的情况下最好使用lru的dict，和多进程冲突

    emb_befor_clean = []
    # pre_cache
    from tqdm import trange
    for idx in trange(len(train_datasets)):
        cur_data = ''.join(train_datasets[idx])
        emb = model.encode(cur_data)
        emb_befor_clean.append(emb)
        emb_cache[idx,:] = emb
    # torch.save(emb_befor_clean, 'emb_befor_clean.pt')

    temp_highdim = np.array(emb_befor_clean)
    x_tsne = TSNE(n_components=2, learning_rate=100, random_state=501).fit_transform(temp_highdim)

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=20)
    plt.legend(fontsize=10)
    plt.title("before clean")
    plt.show()

    # data_idx_chunk = data_idx_list

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


    # calc_similarity(data_idx_chunk)
    num_cpus = cpu_count()
    docs = data_idx_list
    chunk_size = ceil(len(docs) / num_cpus)
    chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
    # 将数据切成cpu个数份
    Parallel(n_jobs=num_cpus)(delayed(calc_similarity)(bit_map=bit_map_f, data_idx_chunk=chunk) for chunk in chunks)
    # 把几个cpu的计算结果拼起来
    torch.save(list(bit_map_f), 'bit_map.pt') # save in case error
    # bit_map = torch.load('bit_map.pt')

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

    for dataset_idx, train_dataset in enumerate(train_datasets_list):

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

    emb_after_clean = []
    # pre_cache
    for idx in range(len(train_datasets)):
        if not bit_map_f[idx]:
            continue
        cur_data = ''.join(train_datasets[idx])
        emb = emb_cache[idx,:]
        emb_after_clean.append(emb)


    temp_highdim = np.array(emb_after_clean)
    x_tsne = TSNE(n_components=2, learning_rate=100, random_state=501).fit_transform(temp_highdim)

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=20)
    plt.legend(fontsize=10)
    plt.title("after clean")
    plt.show()

    print(f"before clean len: {len(train_datasets)}")

    print(f"after clean len: {list(bit_map_f).count(True)}")





