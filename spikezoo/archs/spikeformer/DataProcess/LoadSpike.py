import numpy as np

def load_spike_numpy(path: str) -> (np.ndarray, np.ndarray):
    '''
    Load a spike sequence with it's tag from prepacked `.npz` file.\n
    The sequence is of shape (`length`, `height`, `width`) and tag of
        shape (`height`, `width`).
    '''
    data = np.load(path)
    seq, tag, length = data['seq'], data['tag'], int(data['length'])
    seq = np.array([(seq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return seq, tag

def LoadSpike(path: str) -> (np.ndarray, np.ndarray):
    '''
    Load a spike sequence,  the corresponding ground-truth frame sequence,
    and sequence length.
    spSeq: an ndarray of shape('sequence number', 'height', 'width')
    gtFrames: an ndarray of shape('sequence length', 'height', 'width')
    '''
    data = np.load(path)
    spSeq, gtFrames, length = data['spSeq'], data['gt'], int(data['length'])
    spSeq = np.array([(spSeq[i // 8] >> (i & 7)) & 1 for i in range(length)])
    return spSeq, gtFrames

def load_spike_raw(path: str, width=400, height=250) -> np.ndarray:
    '''
    Load bit-compact raw spike data into an ndarray of shape
        (`sequence length`, `height`, `width`).
    '''
    with open(path, 'rb') as f:
        fbytes = f.read()
    fnum = (len(fbytes) * 8) // (width * height)  # number of frames
    frames = np.frombuffer(fbytes, dtype=np.uint8)
    frames = np.array([frames & (1 << i) for i in range(8)])
    frames = frames.astype(np.bool).astype(np.uint8)
    frames = frames.transpose(1, 0).reshape(fnum, height, width)
    frames = np.flip(frames, 1)
    return frames
