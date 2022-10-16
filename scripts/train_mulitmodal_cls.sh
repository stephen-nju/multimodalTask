export PYTHONPATH=$PYTHONPATH:/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/::/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/task/video_classification
cd /home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/ || exit
export DEFAULT_ROOT_DIR=/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/experiment
# cd "$(dirname "$0")" || exit
# export DIR=$(pwd)/../
# cd ${DIR} || exit
# 切换到工作目录
source activate base
CUDA_VISIABLE_DEVICES=0,1 python task/video_classification/train_multimodal_video_classification.py \
  --default_root_dir="${DEFAULT_ROOT_DIR}" \
  --gpus=2 \
  --max_epochs=2 \
  --strategy=ddp \
  --num_workers=4 \
  --train_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/demo/train.json \
  --test_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/demo/train.json \
  --video_path_prefix=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video/part1_proposal \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model \
  --pretrained_video=/home/nlpbigdata/net_disk_project/zhubin/mmkg/swin_tiny_patch4_window7_224.pth \
  --lr=1e-5 \
  --max_length=256 \
  --batch_size=4
