import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_gemm():
    n = 512
    k = 1024
    m = 256
    dshape1 = (n, k)
    dshape2 = (m, k)
    data1 = relay.var("data1", shape=dshape1)
    data2 = relay.var("data2", shape=dshape2)
    gemm = relay.nn.dense(data1, data2)
    func = relay.Function([data1, data2], relay.Tuple(tvm.runtime.convert([gemm])))
    func = run_opt_pass(func, transform.InferType())

def test_conv():
    batch_size = 1
    input_channel = 3
    h = 224
    w = 224
    output_channel = 64
    kh = 7
    kw = 7
    h_padding = 1
    w_padding = 1
    oh = h + h_padding * 2 - kh + 1
    ow = w + w_padding * 2 - kw + 1
    dshape = (batch_size, input_channel, h, w)
    weight = relay.var("weight", shape=(output_channel, input_channel, kh, kw))
    data = relay.var("data", shape=dshape)
    conv2d = relay.nn.conv2d(
        data, weight, channels=output_channel, kernel_size=(kh, kw), padding=(h_padding, w_padding)
    )
    func = relay.Function([data, weight], relay.Tuple(tvm.runtime.convert([conv2d])))
    func = run_opt_pass(func, transform.InferType())


def test_simple_network():
    batch_size = 1
    dshape = (batch_size, 64, 56, 56)
    weight_conv = relay.var("weight_conv", shape=(64, 64, 3, 3))
    data1 = relay.var("data1", shape=dshape)
    data2 = relay.var("data2", shape=dshape)
    weight_dense = relay.var("weight_dense", shape=(1, 56 * 56 * 64))

    conv2d_1 = relay.nn.conv2d(data1, weight_conv, channels=64, kernel_size=(3, 3), padding=(1, 1))
    conv2d_2 = relay.nn.conv2d(data2, weight_conv, channels=64, kernel_size=(3, 3), padding=(1, 1))
    add = relay.add(conv2d_1, conv2d_2)
    flattened = relay.nn.batch_flatten(add)
    dense_1 = relay.nn.dense(flattened, weight_dense)

    func = relay.Function(
        [data1, data2, weight_conv, weight_dense],
        relay.Tuple(tvm.runtime.convert([conv2d_1, conv2d_2, dense_1, add, flattened])),
    )
    # alter the CONV 2D data layout to test
    func = run_opt_pass(func, transform.AlterOpLayout())


def test_depthwise_conv2d():
    batch_size = 1
    dshape = (batch_size, 64, 56, 56)
    weight_conv = relay.var("weight_depthwiseconv", shape=(64, 1, 3, 3))
    data1 = relay.var("data1", shape=dshape)
    data2 = relay.var("data2", shape=dshape)
    depthwise_conv2d_1 = relay.nn.conv2d(
        data1, weight_conv, kernel_size=(3, 3), padding=(1, 1), groups=64
    )
    depthwise_conv2d_2 = relay.nn.conv2d(
        data2, weight_conv, kernel_size=(3, 3), padding=(1, 1), groups=64
    )
    add = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)
    func = relay.Function(
        [data1, data2, weight_conv],
        relay.Tuple(tvm.runtime.convert([depthwise_conv2d_1, depthwise_conv2d_2, add])),
    )
    func = run_opt_pass(func, transform.InferType())


def test_conv_2d_transpose():
    batch_size = 1
    input_channel = 3
    h = 224
    w = 224
    output_channel = 64
    kh = 7
    kw = 7
    h_padding = 1
    w_padding = 1
    oh = h - h_padding * 2 + kh - 1
    ow = w - w_padding * 2 + kw - 1
    dshape = (batch_size, input_channel, h, w)
    weight = relay.var("weight", shape=(input_channel, output_channel, kh, kw))
    data = relay.var("data", shape=dshape)
    conv2d_transpose = relay.nn.conv2d_transpose(
        data, weight, channels=output_channel, kernel_size=(kh, kw), padding=(h_padding, w_padding)
    )
    func = relay.Function([data, weight], relay.Tuple(tvm.runtime.convert([conv2d_transpose])))
    func = run_opt_pass(func, transform.InferType())