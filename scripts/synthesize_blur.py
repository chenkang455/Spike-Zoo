import os
import cv2
import numpy as np
from tqdm import trange
import spikezoo as sz
import argparse
import shutil


# ? convert the imgs under raw_folder to blur images on blur_folder
# ? Structure as:
# ?  Root
# ?  ├── raw_folder
# ?  ├── blur_folder
# ?  ├── sharp_folder
# ?  └── spike_folder
def process_image(opt, img):
    height, width = img.shape[0], img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # GRAY=0.3*R+0.59*G+0.11*B
    img = img.astype(np.float32) / 255
    if abs(opt.rs_ratio - 1) > 1e-4:
        new_width, new_height = int(width * opt.rs_ratio), int(height * opt.rs_ratio)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return img


# main function
if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_folder",
        type=str,
        default=r"/home/chenkang455/chenk/myproject/SpikeCS/SpikeZoo_github/data/raw_data",
        help="Raw folder for storing the interpolated imgs of the dataset.",
    )
    parser.add_argument("--blur_name", type=str, default=r"blur_data", help="Desired folder name for storing the blurry image files.")
    parser.add_argument("--img_type", type=str, default=".png", help="Input image type.")
    # Assume [0-12] -> blur_6, [13,25] -> blur_19, num_overlap is the length of interpolated frames between 12.png and 13.png,i.e.,12_0.png,...12_6.png
    parser.add_argument("--num_overlap", type=int, default=7, help="Overlap length between two blurry images.")
    parser.add_argument("--num_inter", type=int, default=7, help="Number of interpolated images between two blurry images.")
    parser.add_argument("--num_blur", type=int, default=13, help="Number of images before interpolation to synthesize one blurry frame.")
    parser.add_argument("--num_add", type=int, default=0, help="Additional spike out of the exposure period. Designed for the reblur loss.")
    parser.add_argument("--rs_ratio", type=float, default=1, help="How much to resize the image for simulating .dat files.")

    # config
    opt = parser.parse_args()
    raw_folder = opt.raw_folder
    base_folder = os.path.dirname(opt.raw_folder)
    blur_folder = os.path.join(base_folder, opt.blur_name)
    if os.path.exists(blur_folder):
        shutil.rmtree(blur_folder)
    os.makedirs(blur_folder, exist_ok=True)

    # work
    for dirpath, sub_dirs, sub_files in os.walk(raw_folder):
        if len(sub_files) == 0 or sub_files[0].endswith(opt.img_type) == False:
            continue
        print(f"Processing {dirpath} ing...")
        output_folder = dirpath.replace(raw_folder, blur_folder)
        os.makedirs(output_folder, exist_ok=True)
        sub_files = sorted(sub_files)
        # parameters setting
        num_blur_inter = (opt.num_blur - 1) * (opt.num_inter + 1) + 1  # number of interpolated imgs per blurry one
        str_len = len(sub_files[0].split(".")[0])
        imgs = []
        img_paths = []
        start = 0  # start frame for synthesizing the blurry image
        bais = 0  # bais that overlap between two blurry imgs
        for i in trange(len(sub_files)):
            if i + bais >= len(sub_files):
                break
            file_name = sub_files[i + bais]
            img = cv2.imread(os.path.join(dirpath, file_name))
            img = process_image(opt, img)
            imgs.append(img)
            # synthesize the blurry image during the exposure
            if i % (num_blur_inter) == num_blur_inter - 1:
                end = i + bais
                # skip the first blurry image
                if start == 0:
                    imgs = []
                    bais += opt.num_overlap
                    start = i + bais + 1
                    continue
                # skip the last blurry image
                if end + opt.num_add >= len(sub_files):
                    break
                blurry_image = np.mean(np.stack(imgs, axis=0), axis=0)
                blurry_image = (blurry_image * 255).astype(np.uint8)
                # save
                out_file = sub_files[(start + end) // 2].replace(opt.img_type, ".png")
                cv2.imwrite(os.path.join(blur_folder,out_file), blurry_image)
                print(f"Generating the blurry image {sub_files[(start + end) // 2]} from {sub_files[start]} to {sub_files[end]}")
                imgs = []
                bais += opt.num_overlap
                start = i + bais + 1
