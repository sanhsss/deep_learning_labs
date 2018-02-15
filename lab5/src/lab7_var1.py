import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/sem_a/lab2/mxnet-cu80")

import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 50

train_data = mx.io.ImageRecordIter(
  path_imgrec="../lab2/data/dataset_val.rec",
  data_shape=(3, 128, 128),
  batch_size=batch_size)

test_data = mx.io.ImageRecordIter(
  path_imgrec="../lab2/data/dataset_val.rec",
  data_shape=(3, 128, 128),
  batch_size=batch_size)
print("DATA LOADED")

sym, arg_params, aux_params = mx.model.load_checkpoint('resnext-50', 0)
print("PARAMS LOADED")

model = mx.mod.Module(symbol=sym, context=mx.gpu())
print("MODEL CREATED")

model.fit(train_data, 
        test_data,
        num_epoch=5,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 500),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')

metric = mx.metric.Accuracy()
acc = model.score(test_data, metric)
print(acc)
