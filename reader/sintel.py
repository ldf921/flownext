import os
import re
import skimage.io
from functools import lru_cache
import struct
import numpy as np


def list_data(path):
    dataset = dict()
    pattern = re.compile(r'frame_(\d+).png')
    for part in ('training', 'test'):
        dataset[part] = dict()
        for subset in ('clean', 'final'):
            dataset[part][subset] = []
            for seq in os.listdir(os.path.join(path, part, subset)):
                frames = os.listdir(os.path.join(path, part, subset, seq))
                frames = list(sorted(map(lambda s: int(pattern.match(s).group(1)),
                                     filter(lambda s: pattern.match(s), frames))))
                for i in frames[:-1]:
                    dataset[part][subset].append((
                        os.path.join(path, part, subset, seq, 'frame_{:04d}.png'.format(i)),
                        os.path.join(path, part, subset, seq, 'frame_{:04d}.png'.format(i + 1)),
                        os.path.join(path, part, 'flow', seq, 'frame_{:04d}.flo'.format(i))))
    return dataset


class Flo:
    def __init__(self, w, h):
        self.__floec1__ = float(202021.25)
        self.__floec2__ = int(w)
        self.__floec3__ = int(h)
        self.__floheader__ = struct.pack("fii", self.__floec1__, self.__floec2__, self.__floec3__)
        self.__floheaderlen__ = len(self.__floheader__)
        self.__flow__ = w
        self.__floh__ = h
        self.__floshape__ = [self.__floh__, self.__flow__, 2]

        if self.__floheader__[:4] != b'PIEH':
            raise Exception('Expect machine to be LE.')

    def load(self, file):
        with open(file, 'rb') as fp:
            if fp.read(self.__floheaderlen__) != self.__floheader__:
                raise Exception('Bad flow header: ' + file)
            result = np.ndarray(shape=self.__floshape__,
                                dtype=np.float32,
                                buffer=fp.read(),
                                order='C')
            return result


@lru_cache(maxsize=None)
def load(fname):
    flo = Flo(1024, 436)
    if fname.endswith('png'):
        return skimage.io.imread(fname)
    elif fname.endswith('flo'):
        return flo.load(fname)


sintel_path = r'\\msralab\ProjectData\ehealth02\v-dinliu\Flow2D\Data\Sintel'


if __name__ == '__main__':
    dataset = list_data(sintel_path)
    print(dataset)
    for p in dataset['training']['clean'][0]:
        print(p)
        load(p)
