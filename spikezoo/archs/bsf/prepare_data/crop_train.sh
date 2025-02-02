python3 crop_dataset_train.py -c $1 --eta 1.00 &&
python3 crop_dataset_train.py -c $1 --eta 0.75 &&
python3 crop_dataset_train.py -c $1 --eta 0.50 &&
python3 crop_dataset_train.py --crop_image
