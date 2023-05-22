# data-cleaning-for-llm
数据清洗脚本在 Open-Assistant/model/model_training
运行semantic_deduplication.sh来清洗数据
数据清洗脚本代码是semantic_deduplication.py
通过调整Open-Assistant/model/model_training/configs/config.yaml中的内容来指定要清洗哪些数据
例如在config.yaml中添加以下内容，在semantic_deduplication.sh中指定data_deduplication_zh将清洗其底下指定的4个数据集
```
data_deduplication_zh:
  dataloader_num_workers: 8
  datasets:
    - old_chatglm_alpaca:
        val_split: 0.05
        max_val_set: 250
    - old_chatglm_belle:
        val_split: 0.05
        max_val_set: 250
    - old_chatglm_belle_math:
        val_split: 0.05
        max_val_set: 250
    - old_chatglm_belle_dialog:
        val_split: 0.05
        max_val_set: 250
```
清洗后的数据会被保存在data-cleaning-for-llm/Open-Assistant/model/model_training/cleaned_data/底下
