export NLTK_DATA=/home/nlpbigdata/net_disk_project/zhubin/mmkg/nltk_data
export PYTHONPATH=$PYTHONPATH:/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/
# python task/dense_regression_network/train_drn.py \
# --default_root_dir=/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/logs \
# --gpus=2 \
# --feature_root=/home/nlpbigdata/net_disk_project/zhubin/mmkg/C3D_unit16_overlap0.5_merged \
# --props_file_path=/home/nlpbigdata/net_disk_project/zhubin/mmkg/DRN/data/dataset/Charades/Charades_MAN_32_props.txt \
# --glove_weights=/home/nlpbigdata/net_disk_project/zhubin/mmkg/DRN/data/glove_weights \
# --feature_dim=4096 \
# --ft_window_size=32 \
# --ft_overlap=0.5 \
# --max_epochs=10 \
# --is_first_stage \
# --batch_size=32 \
# --lr=0.001 \



python task/dense_regression_network/train_drn.py \
--gpus=1 \
--default_root_dir=/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/second_stage \
--feature_root=/home/nlpbigdata/net_disk_project/zhubin/mmkg/C3D_unit16_overlap0.5_merged \
--props_file_path=/home/nlpbigdata/net_disk_project/zhubin/mmkg/DRN/data/dataset/Charades/Charades_MAN_32_props.txt \
--glove_weights=/home/nlpbigdata/net_disk_project/zhubin/mmkg/DRN/data/glove_weights \
--feature_dim=4096 \
--ft_window_size=32 \
--ft_overlap=0.5 \
--max_epochs=5 \
--batch_size=2 \
--resume=/home/nlpbigdata/net_disk_project/zhubin/mmkg/CodeRepositoryZhuBin/multimodalTask/logs/lightning_logs/version_0/checkpoints/epoch=9-step=1939.ckpt \
--is_second_stage \
--lr=0.001 \