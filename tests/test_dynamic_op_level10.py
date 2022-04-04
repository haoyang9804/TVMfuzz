import numpy as np
import tvm
from tvm import relay


def test_broadcast_to():
    def verify_more_dynamic_broadcast_to(x_shape, out_shape):
        rank = len(out_shape)
        dtype = "float32"
        shape_type = "int64"
        reshape_shape = relay.Var("shape", relay.ty.TensorType((len(x_shape),), shape_type))
        broadcast_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
        x = relay.Var("x", relay.ty.TensorType((np.prod(x_shape),), dtype))
        r = relay.reshape(x, reshape_shape)
        z = relay.broadcast_to(r, broadcast_shape)

        func = relay.Function([x, reshape_shape, broadcast_shape], z)

        x = np.random.uniform(size=np.prod(x_shape)).astype(dtype)
        ref_res = np.broadcast_to(np.reshape(x, x_shape), out_shape)
        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["vm", "debug"]:
                mod = tvm.ir.IRModule.from_expr(func)
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(
                    x, np.array(x_shape).astype(shape_type), np.array(out_shape).astype(shape_type)
                )


    def verify_broadcast_to(x_shape, out_shape):
        rank = len(out_shape)
        dtype = "float32"
        shape_type = "int64"
        dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
        x = relay.Var("x", relay.ty.TensorType(x_shape, dtype))
        z = relay.broadcast_to(x, dyn_shape)
      
        func = relay.Function([x, dyn_shape], z)


def test_dyn_broadcast_to():
    dtype = "uint8"
    rank = 3
    shape_type = "int64"
    dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
    x_shape = (1,)
    x = relay.Var("x", relay.ty.TensorType(x_shape, dtype))
    z = relay.broadcast_to(x, dyn_shape)
    func = relay.Function([x, dyn_shape], z)


def test_dyn_one_hot():
    def _get_oshape(indices_shape, depth, axis):
        oshape = []
        true_axis = len(indices_shape) if axis == -1 else axis
        ndim = len(indices_shape) + 1
        indices_index = 0
        for i in range(0, ndim):
            if i == true_axis:
                oshape.append(depth)
            else:
                oshape.append(indices_shape[indices_index])
                indices_index += 1

        return oshape

    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        depth_var = relay.var("depth", relay.TensorType((), "int32"))
        on_value_const = relay.const(on_value)
        off_value_const = relay.const(off_value)
        out = relay.one_hot(indices, on_value_const, off_value_const, depth_var, axis, dtype)
        func = relay.Function([indices, depth_var], out)