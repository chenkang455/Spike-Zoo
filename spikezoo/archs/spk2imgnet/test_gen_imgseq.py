import argparse
import time

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.autograd import Variable

from dataset import *
from nets import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Spike_Net_Test")
parser.add_argument(
    "--num_of_layers", type=int, default=17, help="Number of toatal layers"
)
parser.add_argument(
    "--logdir",
    type=str,
    default="./ckpt2/",
    help="path of log files",
)
parser.add_argument("--test_data", type=str, default="./Spk2ImgNet_test2/test2/", help="test set")
parser.add_argument(
    "--save_result", type=bool, default=True, help="save the reconstruction or not"
)
parser.add_argument(
    "--result_dir", type=str, default="results/", help="path of results"
)
parser.add_argument(
    "--exist_gt", type=bool, default=True, help="exist ground truth or not"
)
parser.add_argument("--model_name", type=str, default="model_041.pth", help="Name of ckp")
opt = parser.parse_args()


def normalize(data):
    return data / 255.0


def main():
    # Build model
    print("Loading model ... \n")
    net = SpikeNet(
        in_channels=13, features=64, out_channels=1, win_r=6, win_step=7
    )
    # device_ids = [0]
    # print(device_ids[0])
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.model_name)))
    model.eval()

    # load data info
    print("Loading data info ...\n")
    # sub_dir = 'data4'
    files_source = glob.glob(os.path.join(opt.test_data, "input", "*.dat"))
    files_source.sort()

    # process data
    psnr_test = 0
    ssim_test = 0
    for i in range(len(files_source)):
        sub_dir = files_source[i][:-4]
        # Input spike
        input_f = open(files_source[i], "rb+")
        video_seq = input_f.read()
        video_seq = np.fromstring(video_seq, "B")
        InSpikeArray = raw_to_spike(video_seq, 250, 400)  # c*h*w
        [c, h, w] = InSpikeArray.shape
        for key_id in np.arange(151, 152, 1):
            start_t = time.time()
            SpikeArray = InSpikeArray[key_id - 21 : key_id + 20, :, :]
            # make its shape can be divided by 4
            SpikeArray = np.pad(
                SpikeArray, ((0, 0), (0, 2), (0, 0)), "symmetric"
            )  # c*252*40
            SpikeArray = np.expand_dims(SpikeArray, 0)  # n*c*h*w
            file_name = files_source[i].replace("\\", "/").split("/")[-1]

            SpikeArray = Variable(torch.Tensor(SpikeArray)).cuda()
            with torch.no_grad():
                if opt.exist_gt:
                    out_rec, est0, est1, est2, est3, est4 = model(SpikeArray)
                    out_rec = (
                        torch.clamp(out_rec / 0.6, 0, 1).cpu() * 255
                    )  # 0.6 is the converation rate used in the spike camera. Only neccessary for our synthezed data.
                else:
                    out_rec, est0, est1, est2, est3, est4 = model(SpikeArray)
                    out_rec = torch.clamp(out_rec, 0, 1).cpu() ** (1 / 2.2) * 255
            out_rec = out_rec.detach().numpy().astype(np.float32)
            out_rec = np.squeeze(out_rec).astype(np.uint8)
            # transform to orignal shape # 250*400
            out_rec = out_rec[:250, :]
            if opt.exist_gt:
                gt = cv2.imread(
                    os.path.join(opt.test_data, "gt", file_name[:-3] + "png"), 0
                )
                psnr = peak_signal_noise_ratio(gt, out_rec)
                ssim = structural_similarity(gt, out_rec)
                print("%10s: PSNR:%.2f SSIM:%.4f" % (file_name, psnr, ssim))
                psnr_test += psnr
                ssim_test += ssim
            if opt.save_result:
                if not os.path.exists(os.path.join(opt.result_dir, sub_dir)):
                    os.makedirs(os.path.join(opt.result_dir, sub_dir))
                cv2.imwrite(
                    os.path.join(opt.result_dir, sub_dir, str(key_id) + ".png"), out_rec
                )
                dur_time = time.time() - start_t
                print("dur_time:%.2f", dur_time)

    if opt.exist_gt:
        avg_psnr = psnr_test / len(files_source)
        avg_ssim = ssim_test / len(files_source)
        print("average PSNR: %.2f average SSIM: %.4f" % (avg_psnr, avg_ssim))


if __name__ == "__main__":
    main()
