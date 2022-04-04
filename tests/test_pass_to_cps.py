import numpy as np
import tvm
from tvm import relay
from tvm.relay.transform import to_cps, un_cps
from tvm.relay import create_executor
from tvm.relay import transform


def test_id():
    x = relay.var("x", shape=[])


def test_double():
    t = relay.TypeVar("t")
    x = relay.var("x", t)
    f = relay.var("f", relay.FuncType([t], t))

