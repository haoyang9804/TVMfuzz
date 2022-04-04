import numpy as np

import tvm
from tvm import relay
from tvm.relay import create_executor, transform
import tvm.testing
from tvm.testing import assert_allclose
import pytest


def test_tc():
    """Simple testcase, check that transformation typechecks."""
    mod = tvm.IRModule()

    shape = (20, 20)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x1 = relay.var("x1", t)
    x2 = relay.var("x2", t)
    # f(x1,x2) = (x1-x2)*x2
    y = relay.Function([x1, x2], (x1 - x2) * x2)

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)

    # function input/output types should remain the same
    assert mod["main"].checked_type == relay.FuncType([t, t], t)


def test_add():
    """Simple add testcase. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    # f(x) = x+x
    y = relay.Function([x], x + x)

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    ex = create_executor(mod=mod)


def test_add_tuple():
    """Add elements of tuple. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    tensor_type = relay.TensorType(shape, dtype)
    t = relay.TupleType([tensor_type, tensor_type])

    x = relay.var("x", t)
    # f((x1,x2)) = x1 + x2
    y = relay.Function([x], relay.TupleGetItem(x, 0) + relay.TupleGetItem(x, 1))

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], tensor_type)

    ex = create_executor(mod=mod)


def test_mult():
    """Simple multiplication testcase. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (15, 15)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    # f(x) = x*x
    y = relay.Function([x], x * x)

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    ex = create_executor(mod=mod)


def test_ret_tuple():
    """Test tuple return type. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    # f(x) = (x,x)
    func = relay.Function([x], relay.Tuple([x, x * relay.const(2.0)]))

    mod["main"] = func
    mod = transform.LazyGradientInit()(mod)
    func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], relay.TupleType([t, t]))

    ex = create_executor(mod=mod)


def test_add_broadcast():
    """Test adding matrices of different size. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = "float32"
    t1 = relay.TensorType(shape1, dtype)
    t2 = relay.TensorType(shape2, dtype)

    x1 = relay.var("x1", t1)
    x2 = relay.var("x2", t2)
    func = relay.Function([x1, x2], x1 + x2)

    mod["main"] = func
    mod = transform.LazyGradientInit()(mod)
    func = mod["main"]



def test_reverse_ad_identity():
    """Simple test with reverse mode ad."""
    # of f(x) = x
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)

    func = relay.Function([x], x)
    back_func = transform.gradient(func)

    mod["main"] = back_func
    mod = transform.LazyGradientInit()(mod)
    back_func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType(
        [t], relay.TupleType([t, relay.TupleType([t])])
    )

    ex = create_executor(mod=mod)


def test_multivar_reverse_ad():
    """Simple test with multivariate reverse mode ad."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.var("y", t)

    func = relay.Function([x, y], (x * y) * relay.const(np.ones(shape, dtype)))
    back_func = transform.gradient(func)

    mod["main"] = back_func
    mod = transform.LazyGradientInit()(mod)
    back_func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType(
        [t, t], relay.TupleType([t, relay.TupleType([t, t])])
    )

    ex = create_executor(mod=mod)


def test_partial_eval():
    """Test transformation following reverse mode ad and PartialEval"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    func = relay.Function([], relay.const(np.ones(shape, dtype)))
    back_func = transform.gradient(func)

    mod["main"] = back_func
    back_func = mod["main"]

    transform.PartialEvaluate()(mod)


def test_after_partial_eval():
    """Test transformation following reverse mode ad and PartialEval"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.var("y", t)

    func = relay.Function([x, y], (x * y) * relay.const(np.ones(shape, dtype)))
    back_func = transform.gradient(func)

    mod["main"] = back_func
    back_func = mod["main"]

    seq = tvm.transform.Sequential(
        [transform.PartialEvaluate(), transform.LazyGradientInit(), transform.DeadCodeElimination()]
    )

    mod = seq(mod)

    assert mod["main"].checked_type == relay.FuncType(
        [t, t], relay.TupleType([t, relay.TupleType([t, t])])
    )

    ex = create_executor(mod=mod)


def test_before_partial_eval():
    """Test transformation before PartialEval"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.var("y", t)

    func = relay.Function([x, y], x * y)
    back_func = transform.gradient(func)

    mod["main"] = back_func
    seq = tvm.transform.Sequential(
        [transform.LazyGradientInit(), transform.PartialEvaluate(), transform.DeadCodeElimination()]
    )
    mod = seq(mod)
    back_func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType(
        [t, t], relay.TupleType([t, relay.TupleType([t, t])])
    )

    ex = create_executor(mod=mod)


def test_zeros():
    """Simple test using "zeros" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.zeros(shape, dtype))

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    ex = create_executor(mod=mod)


def test_ones():
    """Simple test using "ones" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.ones(shape, dtype))

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    ex = create_executor(mod=mod)


def test_zeros_like():
    """Simple test using "zeros_like" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.zeros_like(x))

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    ex = create_executor(mod=mod)


def test_ones_like():
    """Simple test using "ones_like" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.ones_like(x))

    mod["main"] = y
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    ex = create_executor(mod=mod)
