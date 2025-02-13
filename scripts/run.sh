# ------------------------ config ------------------------
raw_folder="/home/chenkang455/chenk/myproject/SpikeCS/SpikeZoo_github/data/raw_data"
img_type=".png"
# Desired folder names.
spike_name="spike_data"
blur_name="blur_data"
sharp_name="sharp_data"
# ------------------------ parameters ------------------------
# Overlap length (after interpolation) between two blurry images.
num_overlap=7
# Number of interpolated images between two blurry images #! consistent with the interpolation algorithm!!!
num_inter=7
# Number of images before interpolation to synthesize one spike stream / blurry frame
num_blur=13
# Additional spike out of the exposure period. Designed for the reblur loss.
num_add=20
# Number set to reduce the spike frames.
num_omit=1
# How much to resize the image for simulating .dat files.
rs_ratio=1
# ------------------------ command ------------------------

python simulate_spike.py \
--raw_folder $raw_folder \
--spike_name $spike_name \
--img_type $img_type \
--num_overlap $num_overlap \
--num_inter $num_inter \
--num_blur $num_blur \
--num_add $num_add \
--num_omit $num_omit \
--rs_ratio $rs_ratio

python extract_sharp.py \
--raw_folder $raw_folder \
--sharp_name $sharp_name \
--img_type $img_type \
--num_overlap $num_overlap \
--num_inter $num_inter \
--num_blur $num_blur \
--num_add $num_add \
--rs_ratio $rs_ratio

python synthesize_blur.py \
--raw_folder $raw_folder \
--blur_name $blur_name \
--img_type $img_type \
--num_overlap $num_overlap \
--num_inter $num_inter \
--num_blur $num_blur \
--num_add $num_add \
--rs_ratio $rs_ratio
