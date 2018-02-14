import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/sem_a/lab2/mxnet")

import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)
  
mx.test_utils.download('http://data.mxnet.io/mxnet/models/imagenet/resnext/50-layers/resnext-50-symbol.json')
mx.test_utils.download('http://data.mxnet.io/mxnet/models/imagenet/resnext/50-layers/resnext-50-0000.params')
print("MODEL LOADED")