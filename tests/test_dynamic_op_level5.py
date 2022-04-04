import math
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform


def test_resize_infer_type():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    size = relay.var("size", relay.TensorType((2,), "int8"))
