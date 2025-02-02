import numpy as np
import os
import os.path as osp


def RawToSpike(video_seq, h, w, flipud=True):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)

    return SpikeMatrix

def dat_to_spmat(dat_path,  size):
    f = open(dat_path, 'rb')
    video_seq = f.read()
    video_seq = np.frombuffer(video_seq, 'b')
    sp_mat = RawToSpike(video_seq, size[0], size[1])
    return sp_mat


## Save Raw dat files
def SpikeToRaw(SpikeSeq, save_path):
    """
        SpikeSeq: Numpy array (sfn x h x w)
        save_path: full saving path (string)
    """
    sfn, h, w = SpikeSeq.shape
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        # 模拟相机的倒像
        spike = np.flipud(SpikeSeq[img_id, :, :])
        # numpy按自动按行排，数据也是按行存的
        spike = spike.flatten()
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())

    fid.close()

    return


def save_to_h5(SpikeMatrix, h5path, name):
    f = h5py.File(h5path, 'w')
    f[name] = SpikeMatrix
    f.close()