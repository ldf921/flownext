import os
import re
import numpy as np
from timeit import default_timer

def load(prefix, subset, samples = -1):
    pattern = re.compile('{}(\d+)_(\d+).bin'.format(subset) )
    files = [ (int(pattern.match(f).group(1) ), f) for f in os.listdir(prefix) if pattern.match(f) ]
    files = list(sorted(files))
    ret = []
    for _, f in files:
        # t0 = default_timer()
        n = int(pattern.match(f).group(2) )
        load_batch(os.path.join(prefix, f), n, ret)
        if samples != -1 and len(ret) > samples:
            break
        # print(f, (default_timer() - t0) / n)
    return zip(*ret)

def load_batch(fname, n, ret):
    array_info = [
        (np.uint8, (384, 512, 3), 589824),
        (np.uint8, (384, 512, 3), 589824),
        (np.float32, (384, 512, 2), 1572864)
    ]
    with open(fname, 'rb') as f:
        buffer = f.read()
        offset = 0
        for i in range(0, n):
            arr = []
            for dtype, shape, nbytes in array_info:
                result = np.ndarray(shape=shape,
                    dtype=dtype,
                    buffer=buffer[offset : offset + nbytes],
                    order='C')
                offset += nbytes
                arr.append(result)
            ret.append(arr)

if __name__ == '__main__':
    load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "train")