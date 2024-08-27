import os
import torch

debug = 1

# gpu
ngpu                            = 1
device                          = torch.device('cuda:0')

# generate data
data_root                       = ['dir/train', 'dir/test']
output_root                     = './results/'

image_height                    = 256
image_width                     = 400
batch_size                      = 16
frame_num						            = 10
num_workers                     = 4

# log
model_name					            = './model/'
debug_dir						            = os.path.join('debug')
log_dir                         = os.path.join(debug_dir, 'log')
log_step                        = 10
model_save_root                 = os.path.join(model_name, 'model.pth')
best_model_save_root            = os.path.join(model_name, 'model_best.pth')

# pretrained model path
checkpoint                      = None if not os.path.exists(os.path.join(model_name, 'model.pth')) else os.path.join(model_name, 'model.pth')
start_epoch		                  = 0
start_iter			                = 0

# parameter of train
learning_rate                   = 1e-4
epoch                           = int(1000)

# validation
valid_start_iter				        = 100
valid_step                      = 100

save_epoch                      = 5
eval_epoch                      = 5
