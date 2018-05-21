import argparse
import json
import os
import random
import re
import sys
import time
from timeit import default_timer
import numpy as np
import socket

from reader.chairs import binary_reader, trainval, ppm, flo
from reader import sintel
import mxnet as mx
from mxnet import nd, autograd
import hashlib
import yaml
from reader.client import DatasetClient

model_parser = argparse.ArgumentParser(add_help=False)

training_parser = argparse.ArgumentParser(add_help=False)
training_parser.add_argument('--batch', type=int, default=8, help="minibatch size of samples per device")
training_parser.add_argument('--relative', type=str, default="")

parser = argparse.ArgumentParser(parents=[model_parser, training_parser])

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('-d', '--device', type=str, default='', help="Specify gpu device(s)")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--fake_data', action='store_true')
parser.add_argument('--train_cfg', type=str, default="chairs.yaml")
parser.add_argument('--short_data', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--net_data', action='store_true')
parser.add_argument('--valid', action='store_true', help='Do validation')
parser.add_argument('-c', '--checkpoint', type=str, default=None, 
    help='model heckpoint to load, by default, the latest model is loaded.'
    'You can use checkpoint:steps to load to a specific steps')
parser.add_argument('--tag', type=str, default="")
parser.add_argument('-n', '--network', type=str, default=None)

args = parser.parse_args()

# repoRoot = os.path.dirname(os.path.realpath(__file__))
repoRoot = r'\\msralab\ProjectData\ehealth02\v-dinliu\Flow2D'

if args.device == "":
    ctx = [mx.cpu()]
else:
    ctx = [mx.gpu(gpu_id) for gpu_id in map(int, args.device.split(','))]

import network.config
with open(os.path.join(repoRoot, 'network', 'config', args.train_cfg)) as f:
    train_cfg = network.config.Reader(yaml.load(f))
validation_steps = train_cfg.validation_steps.value
checkpoint_steps = train_cfg.checkpoint_steps.value

def fetch_data(i):
    paths =  ('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm',
            '//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm',
            '//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo')
    return [client.get(path) for path in paths]

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
elif args.net_data:
    print('socket data ...')
    trainSet, validationSet = trainval.read('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/FlyingChairs_train_val.txt')
    client = DatasetClient()
    trainSize = len(trainSet)
    validationSize = len(validationSet)

    validationImg1, validationImg2, validationFlow = zip(*[fetch_data(i) for i in validationSet])
elif train_cfg.dataset.value == 'sintel':
    print('loading sintel dataset ...')
    subsets = train_cfg.subsets.value
    print(subsets[0], subsets[1])
    trainImg1 = []
    trainImg2 = []
    trainFlow = []
    sintel_dataset = sintel.list_data(sintel.sintel_path)
    for k, dataset in sintel_dataset[subsets[0]].items():
        if train_cfg.fast.get(False):
            dataset = dataset[:8]
        img1, img2, flow = [[sintel.load(p) for p in data] for data in zip(*dataset)]
        trainImg1.extend(img1)
        trainImg2.extend(img2)
        trainFlow.extend(flow)
    trainSize = len(trainImg1)

    validationImg1 = []
    validationImg2 = []
    validationFlow = []
    for k, dataset in sintel_dataset[subsets[1]].items():
        if train_cfg.fast.get(False):
            dataset = dataset[:8]
        img1, img2, flow = [[sintel.load(p) for p in data] for data in zip(*dataset)]
        validationImg1.extend(img1)
        validationImg2.extend(img2)
        validationFlow.extend(flow)
    validationSize = len(validationImg1)
else:
    print('reading full data ...')
    t0 = default_timer()
    trainSet, validationSet = trainval.read('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/FlyingChairs_train_val.txt')

    if not args.valid and not args.predict:
        if args.short_data:
            flag = False
            try:
                trainImg1, trainImg2, trainFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "train", 320)
                flag = True
            except:
                print('Fallback loading ...')
            from reader.chairs import ppm, flo
            if not flag:
                trainSet = trainSet[:128]
                trainImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in trainSet]
                trainImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in trainSet]
                trainFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in trainSet]
            trainSize = len(trainImg1)
        else:
            flag = False
            try:
                trainImg1, trainImg2, trainFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "train")
                flag = True
            except:
                print('Fallback loading full train ...')
            if not flag:
                trainImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in trainSet]
                trainImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in trainSet]
                trainFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in trainSet]
            trainSize = len(trainImg1)
    else:
        trainSize = 0

    if not args.predict:
        if args.short_data:
            validationImg1, validationImg2, validationFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "val", 128)
            validationSize = len(validationImg1)
        else:
            validationSize = len(validationSet)
            flag = False
            try:
                validationImg1, validationImg2, validationFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "val")
                flag = True
            except:
                print('Fallback loading full val ...')
            if not flag:
                validationImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in validationSet]
                validationImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in validationSet]
                validationFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in validationSet]
            assert(len(validationImg1) == validationSize)
    else:
        validationSize = 0

    print('Using {}s'.format(default_timer() - t0))

print('data read, train {} val {}'.format(trainSize, validationSize))

import path
import logger
steps = 0
if args.checkpoint is not None:
    if ':' in args.checkpoint:
        prefix, steps = args.checkpoint.split(':')
    else:
        prefix = args.checkpoint
        steps = None
    log_file, run_id = path.find_log(prefix)    
    if steps is None:
        checkpoint, steps = path.find_checkpoints(run_id)[-1]
    else:
        checkpoints = path.find_checkpoints(run_id)
        try:
            checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
        except StopIteration:
            print('The steps not found in checkpoints', steps, checkpoints)
            raise StopIteration
    steps = int(steps)

    _, exp_info = path.read_log(log_file)
    exp_info = exp_info[-1]
    for k in args.__dict__:
        if k in exp_info and k not in ('device', 'valid', 'checkpoint', 'fake_data', 'short_data', 'net_data', 'predict'):
            setattr(args, k, eval(exp_info[k]))
            print('{}={}, '.format(k, exp_info[k]), end='')
    print()
else:
    uid = (socket.gethostname() + logger.FileLog._localtime().strftime('%b%d-%H%M') + args.device)
    if args.tag == "":
        args.tag = hashlib.sha224(uid.encode()).hexdigest()[:3] 
    run_id = args.tag + logger.FileLog._localtime().strftime('%b%d-%H%M')

# import pipeline
from network import get_pipeline
if args.config is not None:
    with open(os.path.join(repoRoot, 'network', 'config', args.config)) as f:
        config = yaml.load(f)
else:
    config = dict()
pipe = get_pipeline(args.network, ctx=ctx, config=config)
lr_schedule = train_cfg.optimizer.learning_rate.get(None)
if lr_schedule is not None:
    pipe.lr_schedule = lr_schedule

if args.checkpoint is not None:
    print('Load Checkpoint {}'.format(checkpoint))
    pipe.load(checkpoint)
    if not args.valid:
        pipe.trainer.step(100, ignore_stale_grad=True)
        pipe.trainer.load_states(checkpoint.replace('params', 'states'))

batch_size = args.batch
assert batch_size % len(ctx) == 0
batch_size_card = args.batch // len(ctx)

if args.valid:
    log = logger.FileLog(os.path.join(repoRoot, 'logs', '{}.val.log'.format(run_id)), screen=True)
else:
    log = logger.FileLog(os.path.join(repoRoot, 'logs', '{}.log'.format(run_id)))
    log.log('start={}, train={}, val={}, host={}, batch={}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size))
    information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
    log.log(information)

if args.predict:
    import valid
    checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
    valid.predict(pipe, os.path.join(repoRoot, 'flows', checkpoint_name), batch_size=args.batch)
    
if args.valid:
    val_epe = pipe.validate(validationImg1, validationImg2, validationFlow, batch_size=args.batch*2)
    log.log('steps={}, chairs.val:epe={}'.format(steps, val_epe))
    val_epe = pipe.validate_levels(validationImg1, validationImg2, validationFlow, batch_size=args.batch*2)
    log.log('steps={}, chairs.val:epe_level={}'.format(steps, val_epe))
    sintel_dataset = sintel.list_data(sintel.sintel_path)
    for div in ('training',):
        for k, dataset in sintel_dataset[div].items():
            img1, img2, flow = [[sintel.load(p) for p in data] for data in zip(*dataset)]
            val_epe = pipe.validate(img1, img2, flow, batch_size=args.batch)
            log.log('steps={}, sintel.{}.{}:epe={}'.format(steps, div, k, val_epe))
    log.close()
    sys.exit(0)



import augmentation
orig_shape = trainImg1[0].shape[:2]
if orig_shape[1] > 800:
    target_shape = (320, 448)
else:
    target_shape = (384, 640)
if args.relative == "":
    aug = augmentation.GeometryAugmentation(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card
                                            )
elif args.relative == "M":
    aug = augmentation.GeometryAugmentation(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.96, 1 / 0.96)
                                            )
elif args.relative == "S":
    aug = augmentation.GeometryAugmentation(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.16, relative_scale=(0.98, 1 / 0.98)
                                            )
color_aug = augmentation.ColorAugmentation(contrast_range=(-0.4, 0.8), brightness_sigma=0.2, channel_range=(0.8, 1.4), batch_size=batch_size_card, 
    shape=target_shape, noise_range=(0, 0.04))
aug.hybridize()
color_aug.hybridize()
def index_generator(n):
    indices = np.arange(0, n, dtype=np.int)
    while True:
        np.random.shuffle(indices)
        yield from indices

train_gen = index_generator(trainSize)
from mxnet import gluon

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
        if args.net_data:
            iq.put(fetch_data(trainSet[i]))
        else:
            iq.put((trainImg1[i], trainImg2[i], trainFlow[i]))

def random_aug(iq, oq):
    while True:
        data = iq.get()
        oq.put([np.transpose(arr, (2, 0, 1)) for arr in data])

def batch_samples(iq, oq, batch_size):
    while True:
        data = []
        for i in range(batch_size):
            data.append(iq.get())
        oq.put([np.stack(x, axis=0) for x in zip(*data)])

def remove_file(iq):
    while True:
        f = iq.get()
        try:
            os.remove(f)
        except OSError as e:
            log.log('Remove failed' + e)

data_queue = Queue(maxsize=100)
aug_queue = Queue(maxsize=100)
batch_queue = Queue(maxsize=4)
remove_queue = Queue(maxsize=50)

def start_daemon(thread):
    thread.daemon = True
    thread.start()

start_daemon(Thread(target=iterate_data, args=(data_queue, train_gen)))
start_daemon(Thread(target=remove_file, args=(remove_queue,)))
for i in range(16):
    start_daemon(Thread(target=random_aug, args=(data_queue, aug_queue)))
for i in range(2):
    start_daemon(Thread(target=batch_samples, args=(aug_queue, batch_queue, batch_size)))

t2 = None
checkpoints = []
while True:
    steps += 1
    if not pipe.set_learning_rate(steps):
        sys.exit(0)
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
    img1, img2, flow = map(lambda arr : nd.array(arr), batch_queue.get())
    loading_time.update(default_timer() - t0)
    epe = pipe.train_batch(img1, img2, flow, color_aug, aug)
    if args.short_data or args.fake_data:
        log.log('debug={}, epe={}, loading_time={:.2f}, total_time={:.2f}'.format(steps, train_epe.average, loading_time.average, total_time.average))
        
    if steps <= 20 or steps % 50 == 0:
        train_epe.update(epe.asscalar()) #pylint: disable=E1101
        log.log('steps={}, epe={}, loading_time={:.2f}, total_time={:.2f}, lr={}'.format(steps, train_epe.average, loading_time.average, total_time.average, pipe.lr))
        # print('Steps {}, epe {}'.format(steps, epe))
        # vecs = nd.reshape(nd.transpose(pred[-1], (0, 2, 3, 1)), (-1, 2))
        # lens = nd.mean(nd.sum(nd.abs(vecs), axis=-1))
        # print(lens.asscalar())

    if args.short_data or args.fake_data:
        if steps == 1:
            prefix = os.path.join(repoRoot, 'weights', 'test_saver')
            pipe.save(prefix)
        if steps >= 20:
            print('Validation')
            val_epe = pipe.validate(validationImg1, validationImg2, validationFlow, batch_size=args.batch)
            log.log('steps={}, val_epe={}'.format(steps, val_epe))
            print('Test finished')
            log.close()
            if steps <= 30:
                os.remove(log._path)
                print('Cleaned Log')
            sys.exit(0)

    if steps % validation_steps == 0:
        if validationSize > 0:
            val_epe = pipe.validate(validationImg1, validationImg2, validationFlow, batch_size=args.batch)
            log.log('steps={}, val_epe={}'.format(steps, val_epe))
        if steps % checkpoint_steps == 0:
            prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
            pipe.save(prefix)
            checkpoints.append(prefix)
            if len(checkpoints) > 3:
                prefix = checkpoints[0]
                checkpoints = checkpoints[1:]
                remove_queue.put(prefix + '.params')
                remove_queue.put(prefix + '.states')
