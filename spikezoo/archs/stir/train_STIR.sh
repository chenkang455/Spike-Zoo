python3 main.py \
--data_root /data/local_userdata/fanbin/REDS_dataset/REDS120fps \
--arch STIR \
--batch_size 8 \
--learning_rate 1e-4 \
--epochs 200 \
--w_per 0.2 \
--milestones 50 100 150 200 \
--lr_scale_factor 0.7 -pf 1