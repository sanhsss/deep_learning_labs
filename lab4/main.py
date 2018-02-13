import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/sem_a/lab2/mxnet-cu80")

import mxnet as mx
import numpy as np
import logging
import time
from autoencoder import pretrain, build_network

logging.getLogger().setLevel(logging.DEBUG)

batch_s = 10

layer_sizes = [5000, 3000, 1000, 800, 50]
pretrain_types=['tanh','relu','tanh','tanh','sigmoid']
train_types=['tanh','relu','tanh','tanh','softmax']
print(layer_sizes)
print(pretrain_types)
print(train_types)
X_train=mx.nd.array(np.load("../lab2/data/train.npy").astype('float32') / 255)

params = pretrain(layer_sizes, pretrain_types, X_train, batch_size=batch_s)
del(params['data'])

train_data = mx.io.ImageRecordIter(
  path_imgrec="../lab2/data/small_dataset_train.rec",
  data_shape=(3, 40, 40),
  batch_size=batch_s)

test_data = mx.io.ImageRecordIter(
  path_imgrec="../lab2/data/small_dataset_val.rec",
  data_shape=(3, 40, 40),
  batch_size=batch_s)

layers = build_network(layer_sizes, train_types)

model = mx.mod.Module(symbol=layers, context=mx.gpu())
t = time.clock()
print ("Training start")
model.fit(train_data,
          eval_data=test_data,
          arg_params=params,
          optimizer='sgd',
          optimizer_params={'learning_rate': 0.01},
          eval_metric='acc',
          batch_end_callback=mx.callback.Speedometer(batch_s, 200),
          num_epoch=1)
print("Clock time difference: %f" % (time.clock() - t))
acc = mx.metric.Accuracy()
model.score(test_data, acc)
print(acc)