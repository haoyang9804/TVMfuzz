import numpy as np
import tvm
from tvm import te
import scipy
from tvm import relay
from tvm.relay import transform
from tvm.contrib.nvcc import have_fp16


def sigmoid(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))


def relu(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy


def rsqrt(x):
    one = np.ones_like(x)
    return one / np.sqrt(x)


def test_unary_op():
    def check_single_op(opfunc, ref, dtype):
        shape = (10, 4)
        dtype = dtype
        tp = relay.TensorType(shape)
        x = relay.var("x", tp, dtype=dtype)
        y = opfunc(x)



@tvm.testing.uses_gpu
def test_binary_op():
    def inst(vars, sh):
        return [vars.get(s, s) for s in sh]

    def check_binary_op(opfunc, ref, dtype):
        # TODO(@jroesch): this piece of code improperly uses type variables.
        n = te.var("n")
        s1 = (5, n, 5)
        s2 = (n, 1)
        t1 = relay.TensorType(s1)
        t2 = relay.TensorType(s2)
        x = relay.var("x", t1, dtype=dtype)
        y = relay.var("y", t2, dtype=dtype)
        z = opfunc(x, y)
        # test printer
        assert ("{}(%x, %y)".format(z.op.name)) in z.astext()
        

        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            t2 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1, dtype=dtype)
            y = relay.var("y", t2, dtype=dtype)
            z = opfunc(x, y)
            x_data = np.random.rand(5, 10, 5).astype(dtype)
            y_data = np.random.rand(5, 10, 5).astype(dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            for target, ctx in tvm.testing.enabled_targets():
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                if (
                    dtype == "float16"
                    and target == "cuda"
                    and not have_fp16(tvm.gpu(0).compute_version)
                ):
                    continue
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)

    for opfunc, ref in [
        (relay.add, np.add),
        (relay.subtract, np.subtract),
        (relay.multiply, np.multiply),
        (relay.divide, np.divide),
        (relay.floor_divide, np.floor_divide),
        (relay.floor_mod, np.fmod),
    ]:
        for dtype in ["float16", "float32"]:
            check_binary_op(opfunc, ref, dtype)


@tvm.testing.uses_gpu
def test_expand_dims():
    # based on topi test
    def verify_expand_dims(dshape, dtype, oshape, axis, num_newaxis):
        x = relay.Var("x", relay.TensorType(dshape, dtype))
        func = relay.Function([x], relay.expand_dims(x, axis, num_newaxis))
        for target, ctx in tvm.testing.enabled_targets():
            if (
                dtype == "float16"
                and target == "cuda"
                and not have_fp16(tvm.gpu(0).compute_version)
            ):
                continue
            data = np.random.uniform(size=dshape).astype(dtype)
            ref_res = data.reshape(oshape)
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(data)



@tvm.testing.uses_gpu
def test_bias_add():
    for dtype in ["float16", "float32"]:
        xshape = (10, 2, 3, 4)
        bshape = (2,)
        rtol = 1e-2 if dtype == "float16" else 1e-5
        x = relay.var("x", shape=xshape, dtype=dtype)
        bias = relay.var("bias", dtype=dtype)
        z = relay.nn.bias_add(x, bias)
        assert zz.args[1].checked_type == relay.TensorType(bshape, dtype)

        func = relay.Function([x, bias], z)
        x_data = np.random.uniform(size=xshape).astype(dtype)
        y_data = np.random.uniform(size=bshape).astype(dtype)
        ref_res = x_data + y_data.reshape((2, 1, 1))
        for target, ctx in tvm.testing.enabled_targets():
            if (
                dtype == "float16"
                and target == "cuda"
                and not have_fp16(tvm.gpu(0).compute_version)
            ):
                continue
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data, y_data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=rtol)


def test_expand_dims_infer_type():
    for dtype in ["float16", "float32"]:
        n, t, d = te.size_var("n"), te.size_var("t"), 100
        x = relay.var("x", shape=(n, t, d), dtype=dtype)
        y = relay.expand_dims(x, axis=2)
        assert "axis=2" in y.astext()


@tvm.testing.uses_gpu
def test_softmax():
    for dtype in ["float16", "float32"]:
        # Softmax accuracy for float16 is poor
        if dtype == "float16":
            return
        shape = (10, 4)
        x = relay.var("x", shape=shape, dtype=dtype)
        y = relay.nn.softmax(x, axis=1)
        assert "nn.softmax" in y.astext()
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=shape).astype(dtype)
        ref_res = tvm.topi.testing.softmax_python(x_data)
        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_log_softmax():
    for dtype in ["float16", "float32"]:
        # Softmax accuracy for float16 is poor
        if dtype == "float16":
            return
        shape = (10, 4)
        x = relay.var("x", shape=shape, dtype=dtype)
        y = relay.nn.log_softmax(x, axis=1)
        assert "nn.log_softmax" in y.astext()
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=shape).astype(dtype)
        ref_res = tvm.topi.testing.log_softmax_python(x_data)
        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_concatenate():
    for dtype in ["float16", "float32"]:
        n, t, d = te.size_var("n"), te.size_var("t"), 100
        x = relay.var("x", shape=(n, t, d))
        y = relay.var("y", shape=(n, t, d))
        z = relay.concatenate((x, y), axis=-1)
        assert "axis=" in z.astext()

        x = relay.exp(x)
        z = relay.concatenate((x, y), axis=2)

        z = relay.concatenate((x, y), axis=1)

        # check shape mismatches (the following case is expected to raise tvm._ffi.base.TVMError.
        try:
            x = relay.var("p1", shape=(2, 5))
            y = relay.var("p2", shape=(2, 3))
            c = relay.concatenate([x, y], axis=0)
            func = relay.Function([x, y], c)
        except tvm._ffi.base.TVMError:
            pass
        else:
            assert False

        x = relay.var("x", shape=(10, 5), dtype=dtype)
        y = relay.var("y", shape=(10, 5), dtype=dtype)
        t = relay.var("z", shape=(), dtype=dtype)
        z = relay.concatenate((x, y), axis=1)
        z = relay.add(z, t)
        # Check result.
        func = relay.Function([x, y, t], z)
        x_data = np.random.rand(10, 5).astype(dtype)
        y_data = np.random.rand(10, 5).astype(dtype)
        t_data = np.random.uniform(size=()).astype(dtype)
        ref_res = np.concatenate((x_data, y_data), axis=1) + t_data

        for target, ctx in tvm.testing.enabled_targets():
            if (
                dtype == "float16"
                and target == "cuda"
                and not have_fp16(tvm.gpu(0).compute_version)
            ):
                continue
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(x_data, y_data, t_data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=0.01)
            op_res2 = intrp2.evaluate(func)(x_data, y_data, t_data)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=0.01)


def test_dropout():
    for dtype in ["float16", "float32"]:
        n, t, d = te.size_var("n"), te.size_var("t"), te.size_var("d")
        input_ty = relay.TensorType((n, t, d), dtype)
        x = relay.var("x", input_ty)
        y = relay.nn.dropout(x, rate=0.75)
        assert "rate=" in y.astext()

def test_batch_norm():
    for dtype in ["float16", "float32"]:
        # beta and gamma ignored
        data = relay.var("data", relay.TensorType((3, 2, 1), dtype))
        beta = relay.var("beta", relay.TensorType((2,), dtype))
        gamma = relay.var("gamma", relay.TensorType((2,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((2,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((2,), dtype))
        y = relay.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, center=False, scale=False
        )


        beta = relay.var("beta", relay.TensorType((3,), dtype))
        gamma = relay.var("gamma", relay.TensorType((3,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((3,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((3,), dtype))

        y = relay.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, axis=0, center=False, scale=False
        )

        # axis=-1
        data = relay.var("data", relay.TensorType((1, 2, 3), dtype))
        beta = relay.var("beta", relay.TensorType((3,), dtype))
        gamma = relay.var("gamma", relay.TensorType((3,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((3,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((3,), dtype))
        y = relay.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, axis=-1, center=False, scale=False
        )


def test_dense_type_check():
    dtype = "float16"
    n, c, h, w = 2, 2, 2, 2
    x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
    # it should fail since it does not match with m(2)
    mismatch_w = 3
    w = relay.var("w", relay.TensorType((2, mismatch_w), dtype))
    y = relay.nn.dense(x, w)


def test_dense():
    for dtype in ["float16", "float32"]:
        # Dense accuracy for float16 is poor
        if dtype == "float16":
            return
        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        w = relay.var("w", relay.TensorType((2, w), dtype))
        y = relay.nn.dense(x, w, units=2)
        assert "units=2" in y.astext()

        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), 2
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        wh, ww = te.size_var("wh"), te.size_var("ww")
        w = relay.var("w", relay.TensorType((ww, wh), dtype))
        y = relay.nn.dense(x, w)

        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), 2
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        w = relay.var("w", relay.IncompleteType())
        y = relay.nn.dense(x, w, units=2)

        x = relay.var("x", shape=(10, 5), dtype=dtype)
        w = relay.var("w", shape=(2, 5), dtype=dtype)
        z = relay.nn.dense(x, w)

        # Check result.
        func = relay.Function([x, w], z)
        x_data = np.random.rand(10, 5).astype(dtype)
        w_data = np.random.rand(2, 5).astype(dtype)
        ref_res = np.dot(x_data, w_data.T)

        for target, ctx in tvm.testing.enabled_targets():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(x_data, w_data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
            op_res2 = intrp2.evaluate(func)(x_data, w_data)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)


def test_dense_dtype():
    data_dtype = "uint8"
    weight_dtype = "int8"
    out_dtype = "uint8"
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), data_dtype))
    w = relay.var("w", relay.TensorType((2, w), weight_dtype))
    y = relay.nn.dense(x, w, units=2, out_dtype=out_dtype)


def test_bitserial_dense():
    m, k = te.size_var("m"), te.size_var("k")
    x = relay.var("x", relay.TensorType((m, k), "int16"))
    w = relay.var("w", relay.TensorType((k, 32), "int16"))
    y = relay.nn.bitserial_dense(x, w, units=32)
    "units=8" in y.astext()

