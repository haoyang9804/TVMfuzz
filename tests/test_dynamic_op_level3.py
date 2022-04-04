import numpy as np
import tvm
from tvm import te
from tvm import relay


def verify_func(func, data, ref_res):
    assert isinstance(data, list)
    for target, ctx in tvm.testing.enabled_targets():
        for kind in ["vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(*data)


@tvm.testing.uses_gpu
def test_dyn_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType((len(newshape),), "int64"))
        z = relay.reshape(x, y)

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        x_data = np.ones(shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)


@tvm.testing.uses_gpu
def test_dyn_shape_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        ref_res = np.reshape(x_data, oshape)


@tvm.testing.uses_gpu
def test_dyn_tile():
    def verify_tile(dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        r = relay.var("reps", relay.TensorType((len(reps),), "float32"))
        z = relay.tile(x, r)

        func = relay.Function([x, r], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)
        reps_data = np.array(reps).astype("float32")


@tvm.testing.uses_gpu
def test_dyn_zeros_ones():
    def verify_zeros_ones(shape, dtype):
        for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
            rank = len(shape)
            dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), "int64"))
            y = op(dyn_shape, dtype)


@tvm.testing.uses_gpu
def test_dyn_full():
    def verify_full(fill_value, src_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        rank = len(src_shape)
        dyn_src_shape = relay.var("dyn_scr_shape", relay.ty.TensorType((rank,), "int64"))
        z = relay.full(x, dyn_src_shape, dtype)
        func = relay.Function([x, dyn_src_shape], z)
        ref_res = np.full(src_shape, fill_value).astype(dtype)


@tvm.testing.uses_gpu
def test_dyn_sparse_to_dense():
    def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
        sparse_indices_data = np.array(sparse_indices)
        sparse_values_data = np.array(sparse_values)
        default_value_data = np.array(default_value)
        output_shape_data = np.array(output_shape)

        a = relay.var(
            "a", relay.TensorType(sparse_indices_data.shape, str(sparse_indices_data.dtype))
        )
        b = relay.var(
            "b", relay.TensorType(sparse_values_data.shape, str(sparse_values_data.dtype))
        )
        output_shape_var = relay.var(
            "output_shape", relay.TensorType(output_shape_data.shape, str(output_shape_data.dtype))
        )
        if default_value is None:
            args = [a, b, output_shape_var]
            d = relay.sparse_to_dense(a, output_shape_var, b)
        else:
            c = relay.var(
                "c", relay.TensorType(default_value_data.shape, str(default_value_data.dtype))
            )
            args = [a, b, c, output_shape_var]
            d = relay.sparse_to_dense(a, output_shape_var, b, c)