import numpy as np
import tvm
from tvm import te

from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_legalize():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def legalize_conv2d(attrs, inputs, types):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            relay.multiply(weight, relay.const(2.0, "float32")),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y



def test_legalize_none():
    """Test doing nothing by returning 'None' """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    def legalize_conv2d(attrs, inputs, types):
        called[0] = True
        return None



def test_legalize_multiple_ops():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def legalize_conv2d(attrs, inputs, types):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def legalize_relu(attrs, inputs, types):
        data = inputs[0]
        add = relay.add(tvm.relay.const(0, "float32"), data)
        return relay.nn.relu(add)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            relay.multiply(weight, relay.const(2.0, "float32")),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.add(tvm.relay.const(0, "float32"), y)
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y



def test_legalize_multi_input():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func

    def legalize_concatenate(attrs, inputs, types):
        # Check that the correct multi-input case is handled.
        assert len(inputs) == 1
        assert isinstance(inputs[0], tvm.relay.expr.Tuple)
        assert len(types) == 2
        assert isinstance(types[0], tvm.relay.ty.TupleType)
        assert isinstance(types[1], tvm.relay.ty.TensorType)
        return None

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func
