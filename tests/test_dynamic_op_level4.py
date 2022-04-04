import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.relay import transform

@tvm.testing.uses_gpu
def test_dynamic_strided_slice():
    def verify(dshape, begin, end, strides, output, slice_mode="end", test_ref=True, dtype="int32"):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)
        if strides:
            if len(strides) == 1:
                strides = strides * ndim
        else:
            strides = [1] * ndim

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(x_data, begin, end, strides, slice_mode)
        data = [x_data, np.array(begin), np.array(end)]

        begin = relay.const(begin, dtype=dtype)
        end = relay.const(end, dtype=dtype)

        if strides:
            data.append(np.array(strides))
            strides = relay.const(strides, dtype=dtype)
            z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
        else:
            z = relay.strided_slice(x, begin=begin, end=end, slice_mode=slice_mode)
        func = relay.Function([x], z)
