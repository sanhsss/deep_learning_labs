import numpy as np

import mxnet as mx
import time

def fist_encoder_metric(label, pred):
    label = label.reshape(-1, 40*40*3)
    return np.mean((label - pred) ** 2)

eval_metric = mx.metric.create(fist_encoder_metric)

def pretrain(layer_sizes, act_types, X_train, batch_size=1):
    params = {}
    encoder_iter = mx.io.NDArrayIter(X_train,
                                     X_train,
                                     batch_size) 
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)
        
    for i, (act_type, num_hidden) in enumerate(zip(act_types, layer_sizes)):
        encoder = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden, name='layer_'+str(i))
        encoder_act = mx.sym.Activation(data=encoder, act_type=act_type, name='layer_'+str(i)+'_act')

        num_hidden_decoder = 40*40*3 if i == 0 else layer_sizes[i-1]
        decoder = mx.symbol.FullyConnected(data=encoder_act, num_hidden=num_hidden_decoder, name='decoder_'+str(i))
        decoder_act = mx.sym.LinearRegressionOutput(data=decoder, name='softmax')
        
        autoencoder = mx.mod.Module(symbol=decoder_act, context=mx.gpu())

        autoencoder.fit(encoder_iter,
                        optimizer='sgd',
                        optimizer_params={'learning_rate': 0.01},
                        eval_metric= (eval_metric if i==0 else 'mse'),
                        batch_end_callback=mx.callback.Speedometer(batch_size, 200),
                        num_epoch=1)
                        
        params.update(autoencoder.get_params()[0].items())
        output = autoencoder.symbol.get_internals()['layer_'+str(i)+'_act_output']
        params.update({'data': X_train}.items())
        
        X_train = output.eval(ctx=mx.cpu(), **params)[0]
        encoder_iter = mx.io.NDArrayIter(X_train,
                                         X_train,
                                         batch_size)
    return params


def build_network(layer_sizes, act_types):
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)

    layer_activation = data
    for i, (act_type, num_hidden) in enumerate(zip(act_types, layer_sizes)):
        layer = mx.symbol.FullyConnected(layer_activation, num_hidden=num_hidden, name='layer_'+str(i))
        
        if act_type == 'softmax':
            layer_activation = mx.sym.SoftmaxOutput(data=layer, name='softmax')
            continue
        
        layer_activation = mx.sym.Activation(data=layer, act_type=act_type, name='layer_'+str(i)+'_act')
    return layer_activation