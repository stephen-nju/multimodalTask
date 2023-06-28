export PYTHONPATH=$PYTHONPATH:/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/::/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/task/video_moment_localization
cd /home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/ || exit
export DEFAULT_ROOT_DIR=/home/nlpbigdata/local_disk/experiment
# cd "$(dirname "$0")" || exit
# export DIR=$(pwd)/../
# cd ${DIR} || exit
# 切换到工作目录
source activate base
python task/video_moment_localization/train_baseline_model_live_video.py \
  --default_root_dir="${DEFAULT_ROOT_DIR}" \
  --gpus=2 \
  --max_epochs=200 \
  --accelerator=gpu \
  --strategy=ddp \
  --num_workers=4 \
  --check_val_every_n_epoch=1 \
  --train_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video_annotation/video_moment_loc_data_v1/train.json \
  --test_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video_annotation/video_moment_loc_data_v1/train.json \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model \
  --lr=1e-3 \
  --max_length=128 \
  --batch_size=16 \
  --optimizer_name=Adam \
  --loss_type=bce \
  --early_stop \
  --patience=20
