"""Test function extraction"""
import tvm
from tvm import relay


def get_conv_net():
    """This gets the net for a case described in fuse_ops.cc:

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |
    """
    dshape = (1, 1, 5, 1)
    x = relay.var("x", shape=dshape)
    y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)

    x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)
    x2 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=1)
    x3 = relay.nn.conv2d(y, relay.var("w4"), kernel_size=(3, 3), padding=(1, 1), channels=1)

    z = relay.add(x1, x2)
    z = relay.add(x3, z)

    return tvm.IRModule.from_expr(z)


def get_conv2d():
    x = relay.var("x", shape=(1, 56, 56, 64))
    weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
    y = relay.nn.conv2d(
        x,
        weight1,
        channels=32,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    return tvm.IRModule.from_expr(y)
