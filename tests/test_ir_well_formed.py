import tvm
from tvm import te
from tvm import relay


def test_let():
    x = relay.Var("x")
    v = relay.Constant(tvm.nd.array(10))
    ty = None
    let = relay.Let(x, v, x)
    f = relay.Function([x], x, ty)


def test_tuple():
    x = relay.Var("x")
    v = relay.Constant(tvm.nd.array(10))
    let = relay.Let(x, v, x)


def test_tuple_get_item():
    t = relay.Var("t")


def test_adt():
    mod = tvm.IRModule()
    x = relay.Var("x")
    some_case = relay.Clause(relay.PatternConstructor(p.some, [relay.PatternVar(x)]), x)
    default_case = relay.Clause(relay.PatternVar(x), x)