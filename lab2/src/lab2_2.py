import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/sem_a/lab2/mxnet-cu80")

import mxnet as mx
import logging

import time

logging.getLogger().setLevel(logging.DEBUG)

batch_s=20
print("IAlex config1")
train_data = mx.io.ImageRecordIter(
  path_imgrec="data/dataset_train.rec",
  data_shape=(3, 128, 128),
  batch_size=batch_s)

test_data = mx.io.ImageRecordIter(
  path_imgrec="data/dataset_val.rec",
  data_shape=(3, 128, 128),
  batch_size=batch_s)
print("data - ok")

data = mx.sym.var('data')
data = mx.sym.flatten(data = data)

layer1 = mx.sym.FullyConnected(data = data, num_hidden=2000)
layer1_act=mx.sym.Activation(data=layer1, act_type='sigmoid')

layer2 = mx.sym.FullyConnected(data = layer1_act, num_hidden=800)
layer2_act=mx.sym.Activation(data=layer2, act_type='sigmoid')

out=mx.sym.FullyConnected(data = layer2_act, num_hidden=15)
out_actsoftmax=mx.sym.SoftmaxOutput(data=out, name='softmax')

print("layers - ok")

fcnn_net = mx.mod.Module(symbol = out_actsoftmax, context = mx.gpu())

print("I'm ready to fit!")
start_time = time.clock()

fcnn_net.fit(train_data,
    eval_data = test_data,
    optimizer = 'sgd',
    optimizer_params = {'learning_rate': 0.001},
    eval_metric = 'acc',
    batch_end_callback = mx.callback.Speedometer(batch_s, 200),
    num_epoch = 1)

fit_time=time.clock()-start_time
print("Fit time: %f", fit_time)

acc = mx.metric.Accuracy()
fcnn_net.score(test_data, acc) 
print(acc)
