import argparse
import json
import os
import random
import re
import sys
import time
from timeit import default_timer
import numpy as np

from reader.chairs import binary_reader, trainval
from reader import sintel
import flownet 
import mxnet as mx
from mxnet import nd, autograd

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=str, default='', help="Specify gpu device(s)")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--fake_data', action='store_true')
parser.add_argument('--short_data', action='store_true')
parser.add_argument('--valid', action='store_true', help='Do validation')
parser.add_argument('--net_args', type=str, default='', help='network arguments')
parser.add_argument('--batch', type=int, default=8, help='size of minibatch')
parser.add_argument('--round', type=int, default=100000, help='number of minibatches per epoch')
parser.add_argument('-c', '--checkpoint', type=str, default=None, help='model heckpoint to load')
args = parser.parse_args()

# repoRoot = os.path.dirname(os.path.realpath(__file__))
repoRoot = r'\\msralab\ProjectData\ehealth02\v-dinliu\Flow2D'

if args.device == "":
    ctx = mx.cpu()
else:
    ctx = mx.gpu(int(args.device))


# load training set and validation set
if args.debug or args.fake_data:
    trainSet = np.arange(0, 128)
    validationSet = np.arange(0, 128)
    trainImg1 = np.random.normal(size=(128, 384, 512, 3))
    trainImg2 = np.random.normal(size=(128, 384, 512, 3))
    trainFlow = np.random.normal(size=(128, 384, 512, 2))
    validationImg1 = np.random.normal(size=(128, 384, 512, 3))
    validationImg2 = np.random.normal(size=(128, 384, 512, 3))
    validationFlow = np.random.normal(size=(128, 384, 512, 2))
    trainSize = validationSize = 128
else:
    print('reading full data ...')
    t0 = default_timer()
    trainSet, validationSet = trainval.read('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/FlyingChairs_train_val.txt')

    if not args.valid:
        if args.short_data:
            trainImg1, trainImg2, trainFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "train", 320)
            trainSize = len(trainImg1)
        else:
            trainSize = len(trainSet)
            trainImg1, trainImg2, trainFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "train")
            assert(len(trainImg1) == trainSize)
    else:
        trainSize = 0

    if args.short_data:
        validationImg1, validationImg2, validationFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "val", 128)
        validationSize = len(validationImg1)
    else:
        validationSize = len(validationSet)
        validationImg1, validationImg2, validationFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "val")
        assert(len(validationImg1) == validationSize)

    print('Using {}s'.format(default_timer() - t0))

print('data read, train {} val {}'.format(trainSize, validationSize))

import augmentation
aug = augmentation.GeometryAugmentation(angle_range=(-17, 17), zoom_range=(0.8, 1), target_shape=(320, 448) )

steps = 0

def index_generator(n):
    indices = np.arange(0, n, dtype=np.int)
    while True:
        np.random.shuffle(indices)
        yield from indices

train_gen = index_generator(trainSize)
from mxnet import gluon
batch_size = args.batch


import pipeline
pipe = pipeline.Pipeline(ctx) 

import logger
run_id = logger.FileLog._localtime().strftime('%b%d-%H%M')
log = logger.FileLog('logs/{}.log'.format(run_id))
log.log('start=1, train={}, val={}'.format(trainSize, validationSize))

class MovingAverage:
    def __init__(self, ratio=0.95):
        self.sum = 0
        self.weight = 1e-8
        self.ratio = ratio

    def update(self, v):
        self.sum = self.sum * self.ratio + v
        self.weight = self.weight * self.ratio + 1

    @property
    def average(self):
        return self.sum / self.weight
    
loading_time = MovingAverage()
total_time = MovingAverage()
train_epe = MovingAverage()

from threading import Thread
from queue import Queue

def iterate_data(iq, gen):
    while True:
        i = next(gen)
        iq.put((trainImg1[i], trainImg2[i], trainFlow[i]))

def random_aug(iq, oq):
    while True:
        data = iq.get()
        oq.put(aug.apply(aug.random_params(), data))

def batch_samples(iq, oq, batch_size):
    while True:
        data = []
        for i in range(batch_size):
            data.append(iq.get())
        oq.put([np.stack(x, axis=0) for x in zip(*data)])

data_queue = Queue(maxsize=100)
aug_queue = Queue(maxsize=100)
batch_queue = Queue(maxsize=4)
Thread(target=iterate_data, args=(data_queue, train_gen)).start()
for i in range(20):
    Thread(target=random_aug, args=(data_queue, aug_queue)).start()
for i in range(2):
    Thread(target=batch_samples, args=(aug_queue, batch_queue, batch_size)).start()

def validate():
    batchEpe = []
    for j in range(0, validationSize, batch_size):
        if j + batch_size <= validationSize:
            batchImg1 = validationImg1[j: j + batch_size]
            batchImg2 = validationImg2[j: j + batch_size]
            batchFlow = validationFlow[j: j + batch_size]

            img1 = nd.array(np.transpose(np.stack(batchImg1, axis=0), (0, 3, 1, 2)) / 255.0, ctx=ctx)
            img2 = nd.array(np.transpose(np.stack(batchImg2, axis=0), (0, 3, 1, 2)) / 255.0, ctx=ctx)
            flow = nd.array(np.transpose(np.stack(batchFlow, axis=0), (0, 3, 1, 2)), ctx=ctx)
            
            batchEpe.append(pipe.validate_batch(img1, img2, flow))
    return np.mean(np.concatenate(batchEpe))

t2 = None
while True:
    steps += 1
    batch = []
    t0 = default_timer()
    if t2:
        total_time.update(t0 - t2)
    t2 = t0
    # for t, i in zip(range(batch_size), train_gen):
    #     batch.append(aug.apply(aug.random_params(), (trainImg1[i], trainImg2[i], trainFlow[i])))
    # log.log('steps={}, qsize={}'.format(steps, aug_queue.qsize()))
    # batch = [ aug_queue.get() for i in range(batch_size) ]
    # img1, img2, flow = [ nd.array(np.stack(x, axis=0), ctx=ctx) for x in zip(*batch) ]
    img1, img2, flow = map(lambda arr : nd.array(arr, ctx=ctx), batch_queue.get())
    loading_time.update(default_timer() - t0)
    epe = pipe.train_batch(img1, img2, flow)
    if steps <= 20 or steps % 50 == 0:
        train_epe.update(epe.asscalar())
        log.log('steps={}, epe={}, loading_time={:.2f}, total_time={:.2f}'.format(steps, train_epe.average, loading_time.average, total_time.average))
        # print('Steps {}, epe {}'.format(steps, epe))
        # vecs = nd.reshape(nd.transpose(pred[-1], (0, 2, 3, 1)), (-1, 2))
        # lens = nd.mean(nd.sum(nd.abs(vecs), axis=-1))
        # print(lens.asscalar())
    if steps % 2500 == 0:
        val_epe = validate()
        log.log('steps={}, val_epe={}'.format(steps, val_epe))
        pipe.network.save_params(os.path.join(repoRoot, 'weights', '{}_{}.params'.format(run_id, steps)))

    if args.debug:
        val_epe = validate()
        log.log('steps={}, val_epe={}'.format(steps, val_epe))
        # network.save_params(os.path.join(repoRoot, 'weights', '{}.params'.format(steps)))        
        break
