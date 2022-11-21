export PYTHONPATH=$PYTHONPATH:/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/::/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/task/video_moment_localization
cd /home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/ || exit
export DEFAULT_ROOT_DIR=/home/nlpbigdata/local_disk/experiment
# cd "$(dirname "$0")" || exit
# export DIR=$(pwd)/../
# cd ${DIR} || exit
# 切换到工作目录
source activate base
python task/video_moment_localization/eval_baseline_model_live_video.py \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model \
  --batch_size=2 \
  --train_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video_annotation/video_moment_loc_data_v1/train.json \
  --test_data=/home/nlpbigdata/net_disk_project/zhubin/mmkg/video_annotation/video_moment_loc_data_v1/train.json \
  --num_workers=4