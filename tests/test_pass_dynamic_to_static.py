import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform


def test_dynamic_to_static_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))


@tvm.testing.uses_gpu
def test_dynamic_to_static_double_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))
        z = relay.reshape(z, relay.shape_of(x))


@tvm.testing.uses_gpu
def test_dynamic_to_static_quad_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z1 = relay.reshape(x, relay.shape_of(y))
        z2 = relay.reshape(z1, relay.shape_of(x))
        z3 = relay.reshape(z2, relay.shape_of(z1))
        z4 = relay.reshape(z3, relay.shape_of(z2))


@tvm.testing.uses_gpu
def test_dynamic_to_static_tile():
    def verify_tile(shape, reps, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(reps, "float32"))
        z = relay.tile(x, relay.shape_of(y))


@tvm.testing.uses_gpu
def test_dynamic_to_static_topk():
    def verify_topk(k, axis, ret_type, is_ascend, dtype):
        shape = (20, 100)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        k_var = relay.const(k)
        out = relay.topk(x, k_var, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x], out)

        np_data = np.random.uniform(size=shape).astype("float32")
        if is_ascend:
            np_indices = np.argsort(np_data, axis=axis)
        else:
            np_indices = np.argsort(-np_data, axis=axis)
        kk = k if k >= 1 else shape[axis]
        if axis == 0:
            np_indices = np_indices[:kk, :]
            np_values = np.zeros(np_indices.shape).astype("float32")
            for i in range(shape[1]):
                np_values[:, i] = np_data[np_indices[:, i], i]
        else:
            np_indices = np_indices[:, :kk]
            np_values = np.zeros(np_indices.shape).astype("float32")
            for i in range(shape[0]):
                np_values[i, :] = np_data[i, np_indices[i, :]]
        np_indices = np_indices.astype(dtype)



@tvm.testing.uses_gpu
def test_dynamic_to_static_broadcast_to():
    def verify_broadcast_to(shape, broadcast_shape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(broadcast_shape, "float32"))
        z = relay.broadcast_to(x, shape=relay.shape_of(y))



@tvm.testing.uses_gpu
def test_dynamic_to_static_zeros_ones():
    def verify_ones_zeros(shape, dtype):
        for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
            x = relay.var("x", relay.TensorType(shape, dtype))
            y = op(relay.shape_of(x), dtype)



@tvm.testing.uses_gpu
def test_dynamic_to_static_resize():
    def verify_resize(shape, scale, method, layout):
        if layout == "NHWC":
            size = (shape[1] * scale, shape[2] * scale)
        else:
            size = (shape[2] * scale, shape[3] * scale)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        size_var = relay.const(np.array(size).astype("float32"))
        coord_trans = "asymmetric" if method == "nearest_neighbor" else "align_corners"
        z = relay.image.resize(
            x, size_var, layout, method, coordinate_transformation_mode=coord_trans
        )

    for method in ["bilinear", "nearest_neighbor"]:
        for layout in ["NCHW", "NHWC"]:
            verify_resize((1, 4, 4, 4), 2, method, layout)


@tvm.testing.uses_gpu
def test_dynamic_to_static_one_hot():
    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        depth_var = relay.const(depth)
        on_value_const = relay.const(on_value)
        off_value_const = relay.const(off_value)
        out = relay.one_hot(indices, on_value_const, off_value_const, depth_var, axis, dtype)
        func = relay.Function([indices], out)



@tvm.testing.uses_gpu
def test_dynamic_to_static_full():
    def verify_full(fill_value, fill_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        y = relay.var("y", relay.TensorType(fill_shape, "int64"))
        z = relay.full(x, relay.shape_of(y), dtype)



def test_dynamic_to_static_upsampling():
    def verify_upsampling(data_shape, scale_h_val, scale_w_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        scale_h = relay.const(scale_h_val)
        scale_w = relay.const(scale_w_val)
        z = relay.nn.upsampling(x, scale_h, scale_w)



def test_dynamic_to_static_upsampling3d():
    def verify_upsampling3d(data_shape, scale_d_val, scale_h_val, scale_w_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        scale_d = relay.const(scale_d_val)
        scale_h = relay.const(scale_h_val)
        scale_w = relay.const(scale_w_val)

        z = relay.nn.upsampling3d(x, scale_d, scale_h, scale_w)



def test_dynamic_to_static_pad():
    def verify_pad(data_shape, pad_width, pad_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        z = relay.nn.pad(x, relay.const(np.array(pad_width)), pad_val)


def test_dynamic_to_static_strided_slice():
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
