from tvm.relay.analysis import get_calibration_data
from tvm.relay.testing import create_workload
from tvm.relay.prelude import Prelude
import tvm.relay as relay
from scipy import special
from tvm.relay import TypeFunctor
from tvm.relay.ty import TupleType
from tvm.relay.analysis import context_analysis
from tvm.relay.analysis import Feature
import scipy
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.testing.synthetic import get_workload
import math
import tvm
from tvm.relay.testing import check_grad
from tvm import nd
from tvm.relay import ExprVisitor
from tvm.relay import create_executor
import tvm.relay.transform
from tvm import relay
import tvm.relay.op as reg
from tvm.relay.adt import TypeData
import sys
from tvm.relay.testing import Prelude
from functools import wraps
from tvm.runtime import container
from tvm import runtime
from tvm.relay.ty import IncompleteType
from tvm.relay.testing import count
import pytest
from tvm.relay import op
from tvm.relay.ty import GlobalTypeVar
import scipy.sparse as sp
from tvm.relay.testing import run_infer_type
import tvm.testing
from tvm.relay.analysis import well_formed
from tvm.relay.op.annotation import compiler_end
import numpy as np
from tvm.relay import analysis
from tvm.ir import structural_equal
import json
from tvm.relay.ty import TensorType
from tvm.relay.transform import un_cps
from tvm.relay.transform import FastMath
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import Any
from tvm.contrib import graph_runtime
from tvm import te
from tvm.relay import TypeMutator
from tvm.relay import TypeVisitor
from tvm.relay.testing import make_nat_expr
from tvm.relay.analysis import check_kind
from tvm.relay.analysis import detect_feature
from tvm.relay.transform import to_cps
from tvm.ir import IRModule
import tvm.relay.transform as _transform
import os
from tvm.relay.ty import RefType
import tvm.relay.testing
from numpy import isclose
from tvm.relay.ty import FuncType
import logging
from tvm.relay.testing import enabled_targets
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay import expr as _expr
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.testing import run_opt_pass
from tvm.relay.ty import TypeRelation
import random
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.ty import TypeCall
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm import relay as rly
from tvm import topi
from tvm.relay import memory_alloc
from tvm.relay.transform import SimplifyInference
from tvm.relay import testing
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.ty import TypeVar
from typing import Union
from tvm.contrib.nvcc import have_fp16
from tvm.relay.backend import compile_engine
import tvm.topi.testing
from tvm.relay.testing import rand
from tvm.relay import transform
from tvm.error import TVMError
from tvm.autotvm.tuner import RandomTuner
from tvm import autotvm
import time
import itertools
from tvm.testing import assert_allclose



def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def test_simplify_reshape():

    def before():
        x = relay.var('x', shape=(1, 16, 16, 16), dtype='float32')
        w = relay.var('w', shape=(32, 16, 3, 3), dtype='float32')
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, (- 1)))
        y = relay.reshape(y, newshape=(4, 8, (- 1), 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, (- 1)))
        return relay.Function([x, w], y)

    def expected():
        x = relay.var('x', shape=(1, 16, 16, 16), dtype='float32')
        w = relay.var('w', shape=(32, 16, 3, 3), dtype='float32')
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(32, 16, 16))
        return relay.Function([x, w], y)

    def symbolic():
        b = tvm.te.size_var('b')
        x = relay.var('x', shape=(b, 16, 16, 16), dtype='float32')
        w = relay.var('w', shape=(32, 16, 3, 3), dtype='float32')
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, (- 1)))
        y = relay.reshape(y, newshape=(4, 8, (- 1), 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, (- 1)))
        return relay.Function([x, w], y)
    z = before()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)
    z = symbolic()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(symbolic(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def before():
    x = relay.var('x', shape=(1, 16, 16, 16), dtype='float32')
    w = relay.var('w', shape=(32, 16, 3, 3), dtype='float32')
    y = relay.nn.conv2d(x, w, padding=(1, 1))
    y = relay.reshape(y, newshape=(1, 16, (- 1)))
    y = relay.reshape(y, newshape=(4, 8, (- 1), 16))
    y = relay.reverse_reshape(y, newshape=(32, 0, (- 1)))
    return relay.Function([x, w], y)


def expected():
    x = relay.var('x', shape=(1, 16, 16, 16), dtype='float32')
    w = relay.var('w', shape=(32, 16, 3, 3), dtype='float32')
    y = relay.nn.conv2d(x, w, padding=(1, 1))
    y = relay.reshape(y, newshape=(32, 16, 16))
    return relay.Function([x, w], y)


def symbolic():
    b = tvm.te.size_var('b')
    x = relay.var('x', shape=(b, 16, 16, 16), dtype='float32')
    w = relay.var('w', shape=(32, 16, 3, 3), dtype='float32')
    y = relay.nn.conv2d(x, w, padding=(1, 1))
    y = relay.reshape(y, newshape=(1, 16, (- 1)))
    y = relay.reshape(y, newshape=(4, 8, (- 1), 16))
    y = relay.reverse_reshape(y, newshape=(32, 0, (- 1)))
    return relay.Function([x, w], y)
zeGMM=symbolic()
UlJcr=transform.InferType()
hSJFq=run_opt_pass(zeGMM,UlJcr)
mlnx8=relay.var('''y''',shape=[],dtype='''uint8''')
rmwLk=relay.const(0,dtype='''int32''')
gh3PS=relay.qnn.op.mul(lhs=mlnx8,rhs=mlnx8,lhs_scale=rmwLk,lhs_zero_point=rmwLk,rhs_scale=rmwLk,rhs_zero_point=rmwLk,output_scale=rmwLk,output_zero_point=rmwLk)
wkW3L=relay.Function([mlnx8,],gh3PS)
asSw9=tvm.IRModule()
jmAUK=relay.GlobalTypeVar('''T''')
mvqOU=relay.TypeVar('''A''')
R8RQ8=relay.TypeData(jmAUK,[mvqOU,],[])
asSw9[jmAUK]=R8RQ8
asSw9['''main''']=wkW3L
riTCk=relay.GlobalVar('''f''')
RcplP=relay.TensorType((1,3,1,),'''int16''')
JaYOf=relay.Var('''x''',RcplP)
QXaDa=JaYOf()
zL76r=relay.var('''v1''')
aObnW=relay.multiply(zL76r,rmwLk)
cReH6=relay.If(QXaDa,aObnW,aObnW)
HkF4W=relay.Function([JaYOf,],cReH6,RcplP,[])
asSw9[riTCk]=HkF4W
asSw9[riTCk]=wkW3L
check_basic_block_normal_form(wkW3L)
jFcDb=analysis.get_total_mac_number(hSJFq)
