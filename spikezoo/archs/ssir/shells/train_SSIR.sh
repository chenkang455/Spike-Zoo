cd ../ &&
python3 main.py \
--data-root your_data_root \
--arch SSIR \
--batch-size 8 \
--learning-rate 4e-4 \
--configs ./configs/SSIR.yml \
--epochs 80 \
--workers 8 \
--w_per 0.2 \
--milestones 20 25 30 35 40 45 50 55 65 70 \
--lr-scale-factor 0.7 -pf 1