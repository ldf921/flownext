import argparse
import json
import os
import random
import re
import sys
import time
from timeit import default_timer

os.environ['PATH'] = r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\bin;' + os.environ['PATH']

import numpy as np
import tensorflow as tf

import tflearn
import visualization
from flownet import FlowNetSimple, Framework
from reader.chairs import binary_reader, trainval
from reader import sintel
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=str, default='0', help="Specify gpu device(s)")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--fake_data', action='store_true')
parser.add_argument('--valid', action='store_true', help='Do validation')
parser.add_argument('--net_args', type=str, default='', help='network arguments')
parser.add_argument('--batch', type=int, default=8, help='size of minibatch')
parser.add_argument('--round', type=int, default=100000, help='number of minibatches per epoch')
parser.add_argument('-c', '--checkpoint', type=str, default=None, help='model heckpoint to load')
args = parser.parse_args()

# repoRoot = os.path.dirname(os.path.realpath(__file__))
repoRoot = r'\\msralab\ProjectData\ehealth02\v-dinliu\Flow2D'

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

net_args = eval('dict({})'.format(args.net_args))
print(net_args)
network = FlowNetSimple('flownet_simple', **net_args)
framework = Framework(devices=len(args.device.split(',')), network=network)
valid_framework = Framework(devices=len(args.device.split(',')), network=network, augmentation=False, train=False)

print('graph built')

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
        # trainImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in trainSet]
        # trainImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in trainSet]
        # trainFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in trainSet]
        trainSize = len(trainSet)
        trainImg1, trainImg2, trainFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "train")
        assert(len(trainImg1) == trainSize)
    else:
        trainSize = 0

    # validationImg1 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img1.ppm') for i in validationSet]
    # validationImg2 = [ppm.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_img2.ppm') for i in validationSet]
    # validationFlow = [flo.load('//msralab/ProjectData/ehealth02/v-dinliu/Flow2D/Data/FlyingChairs_release/data/' + ('%05d' % i) + '_flow.flo') for i in validationSet]
    validationSize = len(validationSet)
    validationImg1, validationImg2, validationFlow = binary_reader.load(r"\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\data\FlyingChairsBlock", "val")
    assert(len(validationImg1) == validationSize)

    print('Using {}s'.format(default_timer() - t0))
print('data read, train {} val {}'.format(trainSize, validationSize))

iterationSize = args.round
batchSize = args.batch
batchImg1 = [None for i in range(0, batchSize)]
batchImg2 = [None for i in range(0, batchSize)]
batchFlow = [None for i in range(0, batchSize)]
batchGamma = np.empty([batchSize], np.float32)
batchCropX = np.empty([batchSize], np.int32)
batchCropY = np.empty([batchSize], np.int32)
batchBrightness = np.empty([batchSize], np.float32)
batchContrast = np.empty([batchSize], np.float32)
indexMap = np.arange(trainSize)

steps = 0
learningRates = [1e-4, 1e-4, 1e-4, 1e-4 / 2, 1e-4 / 4, 1e-4 / 8, 1e-4 / 16, 1e-4 / 32, 1e-4 / 64]


def validate(sess, framework, model=None):
    tflearn.is_training(False, session=sess)
    batchEpe = []
    for j in range(0, validationSize, batchSize):
        if j + batchSize <= validationSize:
            batchImg1 = validationImg1[j: j + batchSize]
            batchImg2 = validationImg2[j: j + batchSize]
            batchFlow = validationFlow[j: j + batchSize]
            # batchGamma = [1.0] * batchSize
            # batchCropX = [32] * batchSize
            # batchCropY = [32] * batchSize
            # batchBrightness = [0.0] * batchSize
            # batchContrast = [1.0] * batchSize

        batchEpe.append(sess.run(framework.epe, framework.feed_dict(
            img1=batchImg1,
            img2=batchImg2,
            flow=batchFlow
        )))

        if model is not None and j == 0:
            batchPred = sess.run(framework.flow, framework.feed_dict(
                img1=batchImg1,
                img2=batchImg2,
                flow=batchFlow
            )) 
            visualization.plot(batchImg1, batchImg2, batchFlow, batchPred, model)

        # batchEpe.append(sess.run(framework.epe, {
        #     'img1:0': batchImg1,
        #     'img2:0': batchImg2,
        #     'flow:0': batchFlow,
        #     'gamma:0': batchGamma,
        #     'cropX:0': batchCropX,
        #     'cropY:0': batchCropY,
        #     'brightness:0': batchBrightness,
        #     'contrast:0': batchContrast
        # }))

    return float(np.mean(batchEpe))


def mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
    return path


def validate_sintel(sess, framework, dataset, path):
    tflearn.is_training(False, session=sess)
    validationImg1, validationImg2, validationFlow = zip(*dataset)
    validationSize = len(dataset)
    batchEpe = []
    c = 0
    for j in tqdm(range(0, validationSize, batchSize)):
        if j + batchSize <= validationSize:
            batchImg1 = [sintel.load(p) for p in validationImg1[j: j + batchSize]]
            batchImg2 = [sintel.load(p) for p in validationImg2[j: j + batchSize]]
            batchFlow = [sintel.load(p) for p in validationFlow[j: j + batchSize]]

        batchEpe.append(sess.run(framework.epe, framework.feed_dict(
            img1=batchImg1,
            img2=batchImg2,
            flow=batchFlow
        )))

        if (j // batchSize) % 5 == 0:
            batchPred = sess.run(framework.flow, framework.feed_dict(
                img1=batchImg1,
                img2=batchImg2,
                flow=batchFlow
            ))
            batchImg1 = batchImg1[::4]
            batchImg2 = batchImg2[::4]
            batchFlow = batchFlow[::4]
            batchPred = batchPred[::4]
            visualization.plot(batchImg1, batchImg2, batchFlow, batchPred, path, c)
            c += len(batchImg1)
    mean_epe = np.mean(batchEpe)
    return float(mean_epe)


saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=5, keep_checkpoint_every_n_hours=5)

if args.valid:
    sintel_valid_framework = Framework(devices=len(args.device.split(',')), 
                                       network=network,
                                       augmentation=False,
                                       train=False,
                                       image_shape=(436, 1024))
    sintel_dataset = sintel.list_data(sintel.sintel_path)
    with tf.Session() as sess:
        # weightsPath = os.path.join(repoRoot, 'weights')
        weightsPath = args.checkpoint
        if os.path.exists(weightsPath + '.meta'):
            saver.restore(sess, weightsPath)
            splits = re.split(r'[/\\]', weightsPath)
            model_name = splits[-2] + '_' + splits[-1]
            path = os.path.join('fig', model_name)
            mkdir(path)
            for k, dataset in sintel_dataset['training'].items():
                print('Dataset Sintel {}, Epe = {:.10f}'.format(
                    k, validate_sintel(sess, sintel_valid_framework, dataset, mkdir(os.path.join(path, k)))))
            print('Dataset Chairs Test, Epe = {:10f}'.format(
                validate(sess, valid_framework, mkdir(os.path.join(path, 'chair')))))
        # models = []
        # for f in os.listdir(weightsPath):
        #     model_name, ext = os.path.splitext(f)
        #     if ext == '.meta':
        #         models.append(model_name)
        # models = list(sorted(models, key=lambda x: int(re.match('model-(\d+)', x).group(1))))
        # results = {}
        # for model in models:
        #     saver.restore(sess, os.path.join(weightsPath, model))
        #     results[model] = validate(sess, valid_framework, os.path.join('fig', model))
        #     print(model, results[model])
        # with open(os.path.join(weightsPath, 'validation.json'), 'w') as fo:
        #     json.dump(results, fo)
    print('done')
    sys.exit(0)

with tf.Session(config=tf.ConfigProto()) as sess:
    tf.global_variables_initializer().run()
    if not args.debug:
        run_id = time.strftime('%b%d-%H%M', time.localtime())
        summaryWriter = tf.summary.FileWriter(os.path.join(repoRoot, 'logs', run_id), sess.graph)
        os.mkdir(os.path.join(repoRoot, 'weights', run_id))
    for lr, ver in zip(learningRates, range(len(learningRates))):
        # Training
        for i in range(0, iterationSize):
            t0 = default_timer()
            np.random.shuffle(indexMap)
            idx = 0
            for j in range(0, batchSize):
                batchImg1[j] = trainImg1[indexMap[idx]]
                batchImg2[j] = trainImg2[indexMap[idx]]
                batchFlow[j] = trainFlow[indexMap[idx]]
                batchGamma[j] = random.uniform(0.7, 1.5)
                batchCropX[j] = random.randint(0, 63)
                batchCropY[j] = random.randint(0, 63)
                batchBrightness[j] = random.uniform(0.0, 0.05)
                batchContrast[j] = random.uniform(0.9, 1.1)
                idx = (idx + 1) % trainSize
            tflearn.is_training(True, session=sess)
            summ, _  = sess.run([framework.summaryOp, framework.adamOpt], {
                'img1:0': batchImg1,
                'img2:0': batchImg2,
                'flow:0': batchFlow,
                'gamma:0': batchGamma,
                'cropX:0': batchCropX,
                'cropY:0': batchCropY,
                'brightness:0': batchBrightness,
                'contrast:0': batchContrast,
                'learningRate:0': lr
            })
            steps += 1
            if steps % 10 == 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                      'Steps: %d' % steps,
                      default_timer() - t0,
                      end='\r')

                if args.debug or steps % 500 == 0:
                    tflearn.is_training(False, session=sess)
                    epe = validate(sess, valid_framework)
                    metrics = {'epe': epe}
                    val_summ = tf.Summary(value=[
                        tf.Summary.Value(tag="val_" + k, simple_value=v) for k, v in metrics.items()
                    ])
                    summaryWriter.add_summary(val_summ, steps)

                if not args.debug:
                    summaryWriter.add_summary(summ, steps)

                if steps % 10000 == 0:
                    saver.save(sess, os.path.join(repoRoot, 'weights', run_id, 'model'), global_step=steps)
print('done')
