# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" Support level3 operator test cases.
"""
import numpy as np
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.relay import create_executor, transform


def test_zeros_ones():
    for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
        y = op(shape=(124, 50), dtype="float64")
        intrp = create_executor()
        intrp_res = intrp.evaluate(y).asnumpy()


def test_unary_identity():
    for op, ref in [
        (relay.zeros_like, np.zeros_like),
        (relay.ones_like, np.ones_like),
        (relay.ceil, np.ceil),
        (relay.floor, np.floor),
        (relay.trunc, np.trunc),
        (relay.round, np.round),
        (relay.abs, np.abs),
        (relay.copy, None),  # np.copy
        (relay.negative, np.negative),
        (relay.sign, np.sign),
    ]:
        shape = (8, 9, 4)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = op(x)
        if ref is not None:
            data = np.random.rand(*shape).astype("float32")
            intrp = create_executor()
            op_res = intrp.evaluate(y, {x: relay.const(data)})
            ref_res = ref(data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)


def test_cast():
    x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
    y = x.astype("int32")
    x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
    y = relay.cast(x, "int32")


def test_clip():
    a = relay.var("a", relay.TensorType((10, 4), "float32"))
    y = relay.clip(a, 1.0, 4.0)

    data = np.random.rand(10, 4).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = np.clip(data, 1.0, 4.0)
    np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)


def test_fixed_point_multiply():
    # Test 23 * 1/16
    # [m,s] = [0.5, -3] = frexp(1/16)
    # M = 0.5*2^31 = 1073741824
    # so M = 1073741824 and s = -3

    a = relay.var("a", relay.TensorType((10, 4), "int32"))
    y = relay.fixed_point_multiply(a, 1073741824, -3)

    data = 23 * np.ones((10, 4)).astype("int32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = np.ones((10, 4)).astype("int32")
    np.testing.assert_allclose(op_res.asnumpy(), ref_res, atol=1)


def test_reinterpret():
    a = relay.var("a", relay.TensorType((1000, 4), "float32"))
    y = relay.reinterpret(a, "int32")

    data = np.random.randn(1000, 4).astype("float32") * 1000
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = data.view("int32")


def test_approximate_transcendental():
    def C(x):
        return relay.expr.const(x, "float32")

    def approx_exp(x):
        # An approximation derived from Opus,
        # https://github.com/xiph/opus/blob/c1c247/celt/mathops.h#L147-L165
        x = relay.minimum(relay.maximum(x, C(-88.0)), C(88.0))
        x = C(127.0) + x * C(1.44269504)
        xf = relay.floor(x)
        i = relay.cast(xf, "int32")
        x = x - xf
        Y = C(0.99992522) + x * (C(0.69583354) + x * (C(0.22606716) + x * C(0.078024523)))
        exponent = relay.left_shift(i, relay.expr.const(23, "int32"))
        exponent = relay.reinterpret(exponent, "float32")
        return exponent * Y

    def approximate_sigmoid(x):
        y = approx_exp(x)
        return y / (y + C(1.0))

    def approximate_tanh(x):
        x = x * C(2.0)
        y = approx_exp(x)
        return (y - C(1.0)) / (y + C(1.0))

    a = relay.var("a", relay.TensorType((1000,), "float32"))
    y = approximate_sigmoid(a)
    data = np.linspace(-5, 5, 1000).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})

    def reference_sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))


    y = approximate_tanh(a)
    data = np.linspace(-5, 5, 1000).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})

    def reference_tanh(x):
        return np.tanh(x)



def test_squeeze():
    def verify_squeeze(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        squeeze = relay.squeeze(x, axis=axis)

        np_axis = tuple(axis) if axis is not None else None

        data = np.random.random_sample(shape).astype(dtype)
        intrp = create_executor()
        op_res = intrp.evaluate(squeeze, {x: relay.const(data)})

def test_transpose_infer_type():
    n, t, d = te.size_var("n"), te.size_var("t"), 100
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.transpose(x, axes=(1, 0, 2))

    y = relay.transpose(x)

def test_transpose():
    def verify_transpose(dshape, axes):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.transpose(x, axes=axes)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.transpose(x_data, axes=axes)


    verify_transpose((2, 3, 4), (0, 2, 1))


def test_squeeze_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(2,))
    assert "axis=" in y.astext()

    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x)
    assert "axis=" not in y.astext()


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_squeeze_bad_axes_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(1,))

def test_reshape_infer_type():
    n, t, d1, d2 = 10, 20, 100, 20
    x = relay.var("x", relay.TensorType((n, t, d1, d2), "float32"))
    y = relay.reshape(x, newshape=(n, t, 2000))
    assert "newshape=" in y.astext()

def test_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reshape(x, newshape=newshape)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)



def test_reshape_like_infer_type():
    # concrete shape
    x = relay.var("x", relay.TensorType((1, 2, 3), "float32"))
    y = relay.var("y", relay.TensorType((1, 6), "float32"))
    z = relay.reshape_like(x, y)

    # symbolic shape
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.var("y", relay.TensorType((1, 8, 8), "float32"))
    z = relay.reshape_like(x, y)


def test_reshape_like():
    def verify_reshape_like(shape, oshape):
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=oshape).astype("float32")
        ref_res = np.reshape(x_data, y_data.shape)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("x", relay.TensorType(oshape, "float32"))
        z = relay.reshape_like(x, y)
        func = relay.Function([x, y], z)




def test_take_infer_type():
    def verify_take(dshape, indices_shape, oshape, axis=None):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        y = relay.take(x, indices, axis=axis)

    d1, d2, d3 = te.var("d1"), te.var("d2"), te.var("d3")
    d4, d5, d6 = te.var("d4"), te.var("d5"), te.var("d6")


def test_take():
    def verify_take(src_shape, indices_src, axis=None, mode="clip"):
        src_dtype = "float32"
        indices_dtype = "int32"
        indices_src = np.array(indices_src, dtype=indices_dtype)
        x = relay.var("x", relay.TensorType(src_shape, src_dtype))
        indices = relay.var("indices", relay.TensorType(indices_src.shape, indices_dtype))
        z = relay.take(x, indices, axis=axis, mode=mode)

        func = relay.Function([x, indices], z)
        x_data = np.random.uniform(low=-1, high=1, size=src_shape).astype(src_dtype)
        np_mode = "raise" if mode == "fast" else mode
        ref_res = np.take(x_data, indices=indices_src, axis=axis, mode=np_mode)




def test_split_infer_type():
    def verify_split(dshape, indices_or_sections, ret_type, axis=None):
        x = relay.var("x", relay.ty.TensorType(dshape, "float32"))
        y = relay.split(x, indices_or_sections, axis=axis)

    idxd = tvm.tir.indexdiv

    d1, d2, d3, d4 = te.var("d1"), te.var("d2"), te.var("d3"), te.var("d4")
    axis = te.var("axis")
    verify_split(
        (5, 5, 2, 2),
        5,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                ]
            )
        ),
        axis=1,
    )
    verify_split(
        (5, 5, 2, 2),
        5,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                ]
            )
        ),
        axis=0,
    )
    verify_split(
        (d1, d2, d3, d4),
        4,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                ]
            )
        ),
        axis=2,
    )
    verify_split(
        (d1, d2, d3, d4),
        2,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((idxd(d1, 2), d2, d3, d4), "float32"),
                    relay.ty.TensorType((idxd(d1, 2), d2, d3, d4), "float32"),
                ]
            )
        ),
        axis=0,
    )
    verify_split(
        (d1, d2, d3, d4),
        (2, 4, 7),
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                    relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                    relay.ty.TensorType((d1, 3, d3, d4), "float32"),
                    relay.ty.TensorType((d1, (d2 - 7), d3, d4), "float32"),
                ]
            )
        ),
        axis=1,
    )


def test_full_infer_type():
    # default settings: match input dtype
    x = relay.var("x", relay.TensorType((), "int8"))
    y = relay.full(x, ())

    # change the shape and dtype
    x = relay.var("x", relay.TensorType((), "float32"))
    y = relay.full(x, (1, 2), "int8")
    "shape=" in y.astext()


def test_full():
    def verify_full(fill_value, src_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        z = relay.full(x, src_shape, dtype)
        func = relay.Function([x], z)
        ref_res = np.full(src_shape, fill_value)

    verify_full(4, (1, 3, 4, 4), "int32")
    # verify_full(4, (1, 3, 4, 4), "int64") # This does not pass, python int32 is not upcast to int64, not sure how to fix it.
    verify_full(4.0, (1, 4), "float32")


def test_full_like_infer_type():
    # concrete shape
    base = relay.var("base", relay.TensorType((1, 2, 3), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)

    # symbolic shape
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    base = relay.var("base", relay.TensorType((n, c, h, w), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)


def test_full_like():
    def verify_full_like(base, fill_value, dtype):
        x_data = np.random.uniform(low=-1, high=1, size=base).astype(dtype)
        x = relay.var("x", relay.TensorType(base, dtype))
        y = relay.var("y", relay.scalar_type(dtype))
        z = relay.full_like(x, y)

        func = relay.Function([x, y], z)
        ref_res = np.full_like(x_data, fill_value)



def test_infer_type_leaky_relu():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.leaky_relu(x, alpha=0.1)
    "alpha=0.1" in y.astext()

    shape = (1, 5, 10, 10)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    z = relay.nn.leaky_relu(x, alpha=0.1)
    assert "alpha=0.1" in z.astext()
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = np.where(x_data > 0, x_data, x_data * 0.1)



def verify_infer_type_prelu(data, alpha, axis, output, dtype="float32"):
    x = relay.var("data", relay.TensorType(data, dtype))
    if alpha:
        y = relay.var("alpha", relay.TensorType(alpha, dtype))
    else:
        y = relay.var("alpha", relay.IncompleteType())
    z = relay.nn.prelu(x, y, axis=axis)

    func = relay.Function([x, y], z)
    x_data = np.random.uniform(low=-1, high=1, size=data).astype(dtype)
    a_data = np.random.uniform(low=-1, high=1, size=alpha).astype(dtype)

    if axis == 1:
        ref_res = (x_data < 0) * (x_data * a_data.reshape(3, 1, 1)) + (x_data >= 0) * x_data
    else:
        ref_res = (x_data < 0) * (x_data * a_data.reshape(1, 1, 3)) + (x_data >= 0) * x_data


def test_arange():
    def verify_arange(start, stop, step):
        dtype = "float32"
        if start is None and step is None:
            x = relay.arange(relay.const(stop, dtype=dtype))
            ref_res = np.arange(stop).astype(dtype)
        elif start is None:
            x = relay.arange(relay.const(stop, dtype=dtype), step=relay.const(step, dtype=dtype))
            ref_res = np.arange(stop, step=step).astype(dtype)
        elif step is None:
            x = relay.arange(relay.const(start, dtype=dtype), relay.const(stop, dtype=dtype))
            ref_res = np.arange(start, stop).astype(dtype)
        else:
            x = relay.arange(
                relay.const(start, dtype=dtype),
                relay.const(stop, dtype=dtype),
                relay.const(step, dtype=dtype),
            )
            ref_res = np.arange(start, stop, step).astype(dtype)

        func = relay.Function([], x)

    verify_arange(None, 20, None)
    verify_arange(None, 20, 2)
    verify_arange(1, 20, None)
    verify_arange(1, 20, 2)
    # arange doesnt' support floating point right now, see type relation
    # verify_arange(1, 20, 1.5)
    verify_arange(1, 20.5, None)
    verify_arange(1, 20, 3)
    verify_arange(20, 1, -1)
    # arange doesnt' support floating point right now, see type relation
    # verify_arange(20, 1, -1.5)


def test_meshgrid():
    def verify_meshgrid(lengths, indexing="ij"):
        input_vars = []
        input_data = []
        for i, length in enumerate(lengths):
            input_name = "x_{}".format(i)
            if length == 0:
                # Scalar
                input_vars.append(relay.var(input_name, relay.scalar_type("float32")))
                input_data.append(np.array(1, "float32"))
            else:
                input_vars.append(relay.var(input_name, relay.TensorType((length,), "float32")))
                input_data.append(np.arange(length).astype("float32"))

        z = relay.meshgrid(input_vars, indexing=indexing).astuple()
        func = relay.Function(input_vars, z)
        # Get ref
        ref_res = np.meshgrid(*input_data, indexing=indexing)


def test_tile():
    def verify_tile(dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.tile(x, reps=reps)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)

    verify_tile((2, 3, 4), (3, 2, 1))
    verify_tile((2, 3, 4), (1, 2))
    verify_tile((2, 3), (3, 2, 1))

def test_repeat():
    def verify_repeat(dshape, repeats, axis):
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        func = relay.Function([x], relay.repeat(x, repeats, axis))
        data = np.random.uniform(size=dshape).astype("float32")
        ref_res = np.repeat(data, repeats, axis)


def test_stack():
    def verify_stack(dshapes, axis):
        y = []
        for shape in dshapes:
            y.append(relay.var("input", relay.TensorType(shape, "float32")))
        x = relay.Tuple(y)
        z = relay.stack(x, axis=axis)

        func = relay.Function(y, z)
        x_data = [np.random.normal(size=shape).astype("float32") for shape in dshapes]
        ref_res = np.stack(x_data, axis=axis)

def test_reverse():
    def verify_reverse(dshape, axis):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.reverse(x, axis=axis)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.flip(x_data, axis)


def test_reverse_sequence():
    def verify_reverse_sequence(x_data, seq_lengths, batch_axis, seq_axis, ref_res):
        seq_lengths_data = np.array(seq_lengths).astype("int32")
        x = relay.var("x", relay.TensorType(x_data.shape, str(x_data.dtype)))
        z = relay.reverse_sequence(x, relay.const(seq_lengths_data), seq_axis, batch_axis)
 
        func = relay.Function([x], z)


def test_scatter():
    def ref_scatter(data, indices, updates, axis=0):
        idx = np.indices(indices.shape).reshape(indices.ndim, -1)

        updated_idx = np.copy(idx)
        indices = indices.reshape(-1)
        for i in range(len(indices)):
            updated_idx[axis, i] = indices[i]
        scattered = np.copy(data)
        scattered[tuple(updated_idx)] = updates[tuple(idx)]
        return scattered

    def verify_scatter(dshape, ishape, axis=0):
        d = relay.var("d", relay.TensorType(dshape, "float32"))
        i = relay.var("i", relay.TensorType(ishape, "int64"))
        u = relay.var("u", relay.TensorType(ishape, "float32"))
        z = relay.op.scatter(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter(data_np, indices_np, updates_np, axis)


def test_scatter_add():
    def ref_scatter_add(data, indices, updates, axis=0):
        output = np.copy(data)
        for index in np.ndindex(*indices.shape):
            new_index = list(index)
            new_index[axis] = indices[index]
            output[tuple(new_index)] += updates[index]
        return output

    def verify_scatter_add(dshape, ishape, axis=0):
        d = relay.var("d", relay.TensorType(dshape, "float32"))
        i = relay.var("i", relay.TensorType(ishape, "int64"))
        u = relay.var("u", relay.TensorType(ishape, "float32"))
        z = relay.op.scatter_add(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter_add(data_np, indices_np, updates_np, axis)
        # TODO(mbrookhart): expand testing when adding more backend schedules

def test_gather():
    def verify_gather(data, axis, indices, ref_res):
        data = np.asarray(data, dtype="float32")
        indices = np.asarray(indices, dtype="int32")
        ref_res = np.asarray(ref_res)

        d = relay.var("x", relay.TensorType(data.shape, "float32"))
        i = relay.var("y", relay.TensorType(indices.shape, "int32"))
        z = relay.gather(d, axis, i)

        func = relay.Function([d, i], z)


def test_gather_nd():
    def verify_gather_nd(xshape, yshape, y_data):
        x = relay.var("x", relay.TensorType(xshape, "float32"))
        y = relay.var("y", relay.TensorType(yshape, "int32"))
        z = relay.gather_nd(x, y)

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(size=xshape).astype("float32")
        ref_res = x_data[tuple(y_data)]

def _verify_infiniteness_ops(relay_op, ref_op):
    for dtype in ["float32", "float16", "float16", "int32", "int16"]:
        shape = (2, 8, 8)
        x = relay.var("x", relay.TensorType(shape, dtype))
        y = relay_op(x)
        
        data = np.random.uniform(size=shape).astype(dtype)
        if dtype.startswith("float"):
            data.ravel()[
                np.random.choice(data.size, int(data.size * 0.5), replace=False)
            ] = np.infty
            data.ravel()[np.random.choice(data.size, int(data.size * 0.5), replace=False)] = np.nan

        intrp = create_executor()
        op_res = intrp.evaluate(y, {x: data})
        ref_res = ref_op(data)


def test_unravel_index():
    def verify_unravel_index(indices, shape, dtype):
        x_data = np.array(indices).astype(dtype)
        y_data = np.array(shape).astype(dtype)
        x = relay.var("x", relay.TensorType(x_data.shape, dtype))
        y = relay.var("y", relay.TensorType(y_data.shape, dtype))

        z = relay.unravel_index(x, y)


        if len(x_data.shape) == 1:
            out_shape = [y_data.shape[0], x_data.shape[0]]
        else:
            out_shape = [y_data.shape[0]]
        assert zz.checked_type == relay.ty.TensorType(out_shape, dtype)

        func = relay.Function([x, y], z)
        ref_res = np.unravel_index(x_data, y_data)
        
def test_sparse_to_dense():
    def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
        sparse_indices_data = np.array(sparse_indices)
        sparse_values_data = np.array(sparse_values)
        default_value_data = np.array(default_value)

        a = relay.var(
            "a", relay.TensorType(sparse_indices_data.shape, str(sparse_indices_data.dtype))
        )
        b = relay.var(
            "b", relay.TensorType(sparse_values_data.shape, str(sparse_values_data.dtype))
        )
        if default_value is None:
            args = [a, b]
            d = relay.sparse_to_dense(a, output_shape, b)
        else:
            c = relay.var(
                "c", relay.TensorType(default_value_data.shape, str(default_value_data.dtype))
            )
            args = [a, b, c]
            d = relay.sparse_to_dense(a, output_shape, b, c)

        func = relay.Function(args, d)


def test_adv_index():
    def verify_adv_index(data_shape, index_shapes):
        dtype = "float32"
        inputs = [relay.var("data", relay.TensorType(data_shape, dtype))]
        np_data = np.random.uniform(size=data_shape).astype(dtype)
        np_indices = []
        for i, index_shape in enumerate(index_shapes):
            limit = data_shape[i]
            np_indices.append(np.random.uniform(0, limit - 1, size=index_shape).astype("int64"))
            inputs.append(relay.var("index_{}".format(i), relay.TensorType(index_shape, "int64")))
        np_out = np_data[tuple(np_indices)]
        np_args = [np_data] + np_indices
        out = relay.op.adv_index(inputs)

        func = relay.Function(inputs, out)