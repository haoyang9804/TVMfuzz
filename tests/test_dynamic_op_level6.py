import numpy as np
import tvm
from tvm import te
from tvm import relay


@tvm.testing.uses_gpu
def test_dynamic_topk():
    def verify_topk(k, axis, ret_type, is_ascend, dtype):
        shape = (20, 100)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        k_var = relay.var("x", relay.TensorType((1,), "float32"))
        out = relay.topk(x, k_var, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x, k_var], out)
