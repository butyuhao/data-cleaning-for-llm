

# export NCCL_IB_DISABLE=1
# export NCCL_IB_HCA=^mlx5

# export NCCL_IB_DISABLE=0
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TC="136"
# export NCCL_IB_SL="5"
# export NCCL_IB_GID_INDEX="3"

cd /data/ecnu/EduChat/Open-Assistant/model/model_training

export PYTHONPATH=$PYTHONPATH:../../oasst-shared

# pip install icecream -i https://pypi.douban.com/simple/

torchrun --standalone --nnodes=1 --nproc_per_node 8 trainer_sft.py --configs educhat_instruct_13b_001 --deepspeed --resume_from_checkpoint