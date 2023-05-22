"""
    These are in the form of 'INSTRUCTION', 'RESPONSE'
"""
from datasets import load_dataset
from torch.utils.data import Dataset

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
MY_INSTRUCTION_DATASETS = {
"old_chatglm_alpaca": "/data/ecnu/data/old_Chatglm_data/old_chatglm_alpaca.json",
"old_chatglm_belle": "/data/ecnu/data/old_Chatglm_data/old_chatglm_Belle.json",
"old_chatglm_belle_math":"/data/ecnu/data/old_Chatglm_data/old_chatglm_BELLE_math.json",
"_alpaca": "/data/ecnu/data/normal_quality/opensource_data/alpaca.jsonl",
"_belle": "/data/ecnu/data/normal_quality/opensource_data/Belle.jsonl",
"_belle_generated_chat": "/data/ecnu/data/normal_quality/opensource_data/BELLE_generated_chat.jsonl",
"_belle_math": "/data/ecnu/data/normal_quality/opensource_data/BELLE_math.jsonl",
"_belle_train_2m": "/data/ecnu/data/normal_quality/opensource_data/BELLE_train_2M.jsonl",
"_coig_single": "/data/ecnu/data/normal_quality/opensource_data/COIG_single.jsonl",
"_firefly": "/data/ecnu/data/normal_quality/opensource_data/firefly.jsonl",
"_flan_cot": "/data/ecnu/data/normal_quality/opensource_data/Flan_CoT.jsonl",
"_instinwild_ch": "/data/ecnu/data/normal_quality/opensource_data/instinwild_ch.jsonl",
"_instinwild_en": "/data/ecnu/data/normal_quality/opensource_data/instinwild_en.jsonl",
"_unnatural_instruction": "/data/ecnu/data/normal_quality/opensource_data/Unnatural_Instruction.jsonl",
"_composition_review": "/data/ecnu/data/normal_quality/zeroshot_data/composition_review.jsonl",
"_correct": "/data/ecnu/data/normal_quality/zeroshot_data/correct.jsonl",
"_poem_generate": "/data/ecnu/data/normal_quality/zeroshot_data/poem_generate.jsonl",
"_poem_transform": "/data/ecnu/data/normal_quality/zeroshot_data/poem_transform.jsonl",
"_reading_comprehension_en": "/data/ecnu/data/normal_quality/zeroshot_data/reading_comprehension_en.jsonl",
"_rewriting": "/data/ecnu/data/normal_quality/zeroshot_data/rewriting.jsonl",
"_story_generate": "/data/ecnu/data/normal_quality/zeroshot_data/story_generate.jsonl",
"_summary": "/data/ecnu/data/normal_quality/zeroshot_data/summary.jsonl",
"_transstyle": "/data/ecnu/data/normal_quality/zeroshot_data/TransStyle.jsonl",
"_writing": "/data/ecnu/data/normal_quality/zeroshot_data/writing.jsonl",
"_writing_cot": "/data/ecnu/data/normal_quality/zeroshot_data/writing_cot.jsonl",
"_chengyu_gushi": "/data/ecnu/data/normal_quality/zeroshot_data/chengyu/chengyu_gushi.jsonl",
"_chengyu_jieshi": "/data/ecnu/data/normal_quality/zeroshot_data/chengyu/chengyu_jieshi.jsonl",
"_chengyu_laiyuan": "/data/ecnu/data/normal_quality/zeroshot_data/chengyu/chengyu_laiyuan.jsonl",
"_chengyu_liju": "/data/ecnu/data/normal_quality/zeroshot_data/chengyu/chengyu_liju.jsonl",
"_chengyu_pinyin": "/data/ecnu/data/normal_quality/zeroshot_data/chengyu/chengyu_pinyin.jsonl",
"_jieshi_chengyu": "/data/ecnu/data/normal_quality/zeroshot_data/chengyu/jieshi_chengyu.jsonl",
"_biaoti_chaodai": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/biaoti_chaodai.jsonl",
"_biaoti_gushi": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/biaoti_gushi.jsonl",
"_biaoti_leixing": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/biaoti_leixing.jsonl",
"_biaoti_mingju": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/biaoti_mingju.jsonl",
"_biaoti_zuozhe": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/biaoti_zuozhe.jsonl",
"_gushi-beijing_yiwen": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi-beijing_yiwen.jsonl",
"_gushi_beijing": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_beijing.jsonl",
"_gushi_biaoti": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_biaoti.jsonl",
"_gushi_chaodai": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_chaodai.jsonl",
"_gushi_gushi": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_gushi.jsonl",
"_gushi_leixing": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_leixing.jsonl",
"_gushi_mingju": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_mingju.jsonl",
"_gushi_shangxi": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_shangxi.jsonl",
"_gushi_yiwen": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_yiwen.jsonl",
"_gushi_zuozhe": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/gushi_zuozhe.jsonl",
"_leixing_biaoti": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/leixing_biaoti.jsonl",
"_leixing_gushi": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/leixing_gushi.jsonl",
"_leixing_zuozhe": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/leixing_zuozhe.jsonl",
"_mingju_biaoti-yuanwen": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/mingju_biaoti-yuanwen.jsonl",
"_mingju_zuozhe": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/mingju_zuozhe.jsonl",
"_shiren_chaodai": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/shiren_chaodai.jsonl",
"_shiren_jieshao": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/shiren_jieshao.jsonl",
"_zuozhe-leixing_biaoti": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/zuozhe-leixing_biaoti.jsonl",
"_zuozhe_gushi": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/zuozhe_gushi.jsonl",
"_zuozhe_leixing": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/zuozhe_leixing.jsonl",
"_zuozhe_mingju": "/data/ecnu/data/normal_quality/zeroshot_data/gushi/zuozhe_mingju.jsonl",
"_daan_miyu": "/data/ecnu/data/normal_quality/zeroshot_data/miyu/daan_miyu.jsonl",
"_miyu_daan": "/data/ecnu/data/normal_quality/zeroshot_data/miyu/miyu_daan.jsonl",
}

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
