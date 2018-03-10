from reader.chairs import trainval, ppm, flo
from timeit import default_timer
import random
import numpy as np
import sys
import os

def pack_data(fname, train_imgs1, train_imgs2, flows):
    with open(fname, 'wb') as f:
        z = 0
        for a, b, c in zip(train_imgs1, train_imgs2, flows):
            # if z == 0:
            #     z = 1
            #     print(a.dtype, b.dtype, c.dtype, a.shape, b.shape, c.shape, [a.nbytes, b.nbytes, c.nbytes])
            f.write(a.tobytes() )
            f.write(b.tobytes() )
            f.write(c.tobytes() )
    
def load_pack_data():
    array_info = [
        (np.uint8, (384, 512, 3), 589824, ),
        (np.uint8, (384, 512, 3), 589824, ),
        (np.float32, (384, 512, 2), 1572864)
    ]
    with open('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/pack.bin', 'rb') as f:
        buffer = f.read()
        ret = []
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
        return ret

def pack_all():
    trainSet, validationSet = trainval.read('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/FlyingChairs_train_val.txt')
    
    n = 64
    for name, Set in [ ('train', trainSet), ('val', validationSet) ]:
        bn = 0
        for i in range(0, len(Set), n):
            subset = Set[i:i+n]
            trainImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in subset]
            trainImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in subset]
            trainFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in subset]
            prefix = r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock'
            pack_data(os.path.join(prefix, '{}{}_{}.bin'.format(name, bn, len(subset) ) ), trainImg1, trainImg2, trainFlow)
            bn += 1
            print('{}/{}'.format(i, len(Set)))

pack_all()
sys.exit(0)

trainSet, validationSet = trainval.read('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/FlyingChairs_train_val.txt')
n = 50

T0 = default_timer()
arr = load_pack_data()
print( ((default_timer() - T0) / n) )

# trainSet=random.sample(trainSet, n)
# s = random.randint(1000, 20000)
s = 10000
trainSet=trainSet[s:s+n]
T0 = default_timer()
trainImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in trainSet]
trainImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in trainSet]
trainFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in trainSet]
trainSize = len(trainSet)
print( (default_timer() - T0) / n, s )

# T0 = default_timer()
# pack_data(trainImg1, trainImg2, trainFlow)
# print( (default_timer() - T0) / n)

# for i in range(0, n):
#     z1 = np.all(arr[i][0] == trainImg1[i]) 
#     z2 = np.all(arr[i][1] == trainImg2[i]) 
#     z3 = np.all(arr[i][2] == trainFlow[i]) 
#     print(z1, z2, z3)