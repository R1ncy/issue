#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from datautils import preprocess, dataiterator
import mxnet as mx
import numpy as np
from mxmodel import model_param, kmax_pooling
from metric import mxmetric

logging.basicConfig(level=logging.DEBUG)
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


def fold(x, shape):
    if int(shape[3]) % 2 != 0:
        pad_width = (0, 0, 0, 0, 0, 0, 0, 1)
        x = mx.sym.Pad(data=x, mode='edge', pad_width=pad_width)
    long_rows = mx.sym.Reshape(data=x, shape=(int(shape[0]), int(shape[1]), -1, 2))
    sumed = mx.sym.sum(long_rows, axis=3, keepdims=True)
    fold_out = mx.sym.Reshape(data=sumed, shape=(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3] / 2)))
    return fold_out


def get_dcnn(sentence_size, embed_size, batch_size, vocab_size,
             dropout=0.5, ktop=4, filter_widths=[5, 3],
             filters=[8, 16], conv_wds=[0.000015, 0.0000015]):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    embed_layer = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=embed_size, name='embed',
                                   attr={'wd_mult': '0.005'})
    embed_out = mx.sym.Reshape(data=embed_layer, shape=(batch_size, 1, sentence_size, embed_size))
    layers = [embed_out]
    
    nl = float(len(filters))
    for i in range(len(filters)):
        
        conv_outi = mx.sym.Convolution(data=layers[-1], name="conv%s" % i, kernel=(filter_widths[i], 1),
                                       num_filter=filters[i], pad=(filter_widths[i] - 1, 0),
                                       attr={'wd_mult': str(conv_wds[i])}, no_bias=True)
        _, out_shape, _ = conv_outi.infer_shape(data=(batch_size, sentence_size))
        fold_outi = fold(conv_outi, out_shape[0])

        ki = ktop if i == nl - 1 else max(ktop, int(np.ceil((nl - i - 1) / nl * float(out_shape[0][2]))))

        pool_outi = mx.symbol.Custom(data=fold_outi, name='k_max_pool%s' % i, op_type='k_max_pool', k=ki)
        act_outi = mx.sym.Activation(data=pool_outi, act_type='tanh', name="act%s" % i)
        layers.append(act_outi)

    if dropout > 0.0:
        dp_out = mx.sym.Dropout(data=layers[-1], p=dropout, name="dp")
    else:
        dp_out = layers[-1]

    fc = mx.symbol.FullyConnected(data=dp_out, num_hidden=2, name='fc', attr={'wd_mult': '0.005'})
    dcnn = mx.sym.SoftmaxOutput(data=fc, label=label, name='softmax')
    return dcnn


def train_model(args, ctx):

    logger = logging.getLogger()
    fh = logging.FileHandler('./log/road_all2.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    print('Loading data ..')
    train_path = '../dataset/road_all2_train.xlsx'
    test_path = '../dataset/road_all2_test.xlsx'
    vocab_path = '../dataset/road_all2_vocab'

    train_x, train_y, train_len, vocab = preprocess.get_mx_data(train_path, vocab_path)
    test_x, test_y, test_len, _ = preprocess.get_mx_data(test_path, vocab_path)

    vocab_size = len(vocab)
    print(vocab_size)
    kv = mx.kvstore.create(args.kv_store)

    print("Building iterator ..")
    trainiter = dataiterator.DataIterator(train_x, train_y, train_len, args.batch_size)
    testiter = dataiterator.DataIterator(test_x, test_y, test_len, args.batch_size)

    def sym_gen(seq_len):
        sym = get_dcnn(seq_len, args.embed_size, args.batch_size, vocab_size)
        data_name = ['data']
        label_name = ['label']
        return sym, data_name, label_name

    mod = mx.mod.BucketingModule(sym_gen, context=ctx,
                                 default_bucket_key=trainiter.default_bucket_key)

    arg_params = None
    optimizer_params = {'wd': 0.005} 
    model_prefix = args.prefix + "-%d" % (kv.rank)
    epoch_end_callback = mx.callback.do_checkpoint(model_prefix, period=1)

    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(mx.metric.F1())
    eval_metric.add(mx.metric.CrossEntropy())

    print("start training...")
    mod.fit(trainiter, eval_data=testiter, eval_metric=eval_metric,
            epoch_end_callback=epoch_end_callback,
            kvstore=kv, optimizer='adadelta',
            optimizer_params=optimizer_params,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            arg_params=arg_params, allow_missing=True,
            begin_epoch=args.begin_epoch,
            num_epoch=args.num_epoch)

    print("Train done for epoch: %s" % args.num_epoch)


if __name__ == '__main__':
    mx.random.seed(1301)
    np.random.seed(1301)
    #ctx = mx.cpu(0)
    ctx = mx.gpu()
    args = model_param.parse_args()
    train_model(args, ctx)

