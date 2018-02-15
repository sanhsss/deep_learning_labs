import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/sem_a/lab2/mxnet-cu80")

import numpy as np
import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)

train= mx.io.ImageRecordIter(
	path_imgrec="data/small_dataset_train.rec",
	data_shape=(3, 40, 40),
	batch_size=10)

X_train=[]

for batch in train:
	for item in batch.data[0]:
		img=item.asnumpy()
		X_train.append(img)
print(X_train)
np.save("data/train",X_train) 

