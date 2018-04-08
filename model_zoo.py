import mxnet as mx
model = mx.gluon.model_zoo.vision.resnet50_v1(pretrained=True, root=r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\.mxnet\models', ctx=mx.gpu(0))
print(model(mx.nd.random_uniform(shape=[1, 3, 224, 224]).as_in_context(mx.gpu(0))))