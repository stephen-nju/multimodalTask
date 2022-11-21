export PYTHONPATH=$PYTHONPATH:/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/::/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/task/video_classification
cd /home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/ || exit
export DEFAULT_ROOT_DIR=/home/nlpbigdata/local_disk/experiment
# cd "$(dirname "$0")" || exit
# export DIR=$(pwd)/../
# cd ${DIR} || exit
# 切换到工作目录
source activate base
CUDA_VISIABLE_DEVICES=0,1 python task/video_classification/train_multimodal_video_classification.py \
  --default_root_dir="${DEFAULT_ROOT_DIR}" \
  --gpus=1 \
  --max_epochs=4 \
  --accelerator=gpu \
  --strategy=ddp \
  --num_workers=4 \
  --check_val_every_n_epoch=2 \
  --train_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video_annotation/part1_proposal_ann/train_multimodal.json \
  --test_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video_annotation/part1_proposal_ann/test_multimodal.json \
  --video_path_prefix=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video/part1_proposal \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model \
  --pretrained_video_model=/home/nlpbigdata/net_disk_project/zhubin/mmkg/swin_tiny_patch4_window7_224.pth \
  --lr=1e-3 \
  --max_length=128 \
  --batch_size=8 \
  --accumulate_grad_batches=2 \
  --optimizer_name=SGD \
  --momentum=0.9 \
