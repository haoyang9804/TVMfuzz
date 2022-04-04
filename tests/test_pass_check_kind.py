import tvm
from tvm import te
from tvm import relay
import pytest

def test_typevar_kind():
    # returns the same kind
    tp1 = relay.TypeVar("tp1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.ShapeVar)
    tp3 = relay.TypeVar("tp3", relay.TypeKind.Constraint)



def test_tuple_kind():
    # only contain type kinds
    tp = relay.TypeVar("tp", relay.TypeKind.Type)
    tt = relay.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")
    tf = relay.FuncType(
        tvm.runtime.convert([]), tt, tvm.runtime.convert([]), tvm.runtime.convert([])
    )
    fields = tvm.runtime.convert([tp, tf, tt])

    tup_ty = relay.TupleType(fields)


def test_func_kind():
    # only contain type kinds
    tp1 = relay.TypeVar("tp1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.Type)

    shape = tvm.runtime.convert([1, 2, 3])
    dtype = "float32"
    tensor_type = relay.TensorType(shape, dtype)

    tr = relay.TypeRelation(None, tvm.runtime.convert([tensor_type, tp1]), 1, None)

    type_params = tvm.runtime.convert([tp1, tp2])
    type_constraints = tvm.runtime.convert([tr])
    arg_types = tvm.runtime.convert([tp1, tensor_type])
    ret_type = relay.TupleType(tvm.runtime.convert([tp2, tensor_type]))

    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)


def test_ref_kind():
    # only contain type kinds
    tt = relay.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")
    ft = relay.FuncType(
        tvm.runtime.convert([]), tt, tvm.runtime.convert([]), tvm.runtime.convert([])
    )

    rt1 = relay.RefType(tt)
    rt2 = relay.RefType(ft)
    rt3 = relay.RefType(relay.TupleType([rt1, rt2]))


def test_relation_kind():
    # only have type kinds for arguments
    tp = relay.TypeVar("tp", relay.TypeKind.Type)
    tt = relay.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")
    tf = relay.FuncType(
        tvm.runtime.convert([]), tt, tvm.runtime.convert([]), tvm.runtime.convert([])
    )
    args = tvm.runtime.convert([tf, tt, tp])

    tr = relay.TypeRelation(None, args, 2, None)


def test_global_typevar_kind():
    v1 = relay.GlobalTypeVar("gtv1", relay.TypeKind.AdtHandle)
    v2 = relay.GlobalTypeVar("gtv2", relay.TypeKind.Type)



def test_typecall_kind():
    gtv = relay.GlobalTypeVar("gtv")

    mod = tvm.IRModule()
    data = relay.TypeData(gtv, [], [])
    mod[gtv] = data
    empty_call = relay.TypeCall(gtv, [])

    new_mod = tvm.IRModule()
    tv = relay.TypeVar("tv")
    new_data = relay.TypeData(gtv, [tv], [])
    new_mod[gtv] = new_data
    call = relay.TypeCall(gtv, [relay.TupleType([])])


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_invalid_tuple_kind():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.ShapeVar)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.BaseType)
    tp3 = relay.TypeVar("tp3", relay.TypeKind.Constraint)
    fields = tvm.runtime.convert([tp1, tp2, tp3])

    tup_ty = relay.TupleType(fields)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_invalid_func_kind():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.ShapeVar)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.BaseType)
    tp3 = relay.TypeVar("tp3", relay.TypeKind.Constraint)

    type_params = tvm.runtime.convert([tp1, tp2, tp3])
    type_constraints = tvm.runtime.convert([])
    arg_types = tvm.runtime.convert([tp1, tp2])
    ret_type = tp3

    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_invalid_ref_kind():
    tp = relay.TypeVar("tp", relay.TypeKind.ShapeVar)
    rt = relay.RefType(tp)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_invalid_relation_kind():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.ShapeVar)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.BaseType)
    tp3 = relay.TypeVar("tp3", relay.TypeKind.Constraint)
    args = tvm.runtime.convert([tp1, tp2, tp3])

    func = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Broadcast")
    tr = relay.TypeRelation(func, args, 2, None)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_typecall_invalid_callee():
    # global type var must be an ADT handle
    gtv = relay.GlobalTypeVar("v1", relay.TypeKind.Type)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_typecall_invalid_args():
    # args must all be type kind
    mod = tvm.IRModule()
    gtv = relay.GlobalTypeVar("v1")
    data = relay.TypeData(gtv, [], [])
    mod[gtv] = data



@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_typecall_invalid_num_args():
    mod = tvm.IRModule()
    gtv = relay.GlobalTypeVar("v1")
    tv = relay.TypeVar("tv")
    data = relay.TypeData(gtv, [tv], [])
    mod[gtv] = data


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_func_with_invalid_ret_type():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.ShapeVar)
    tf = relay.FuncType(
        tvm.runtime.convert([tp1]), tp2, tvm.runtime.convert([tp1, tp2]), tvm.runtime.convert([])
    )


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_func_with_invalid_arg_types():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.ShapeVar)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.Type)
    tf = relay.FuncType(
        tvm.runtime.convert([tp1]), tp2, tvm.runtime.convert([tp1, tp2]), tvm.runtime.convert([])
    )


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_func_with_invalid_tuple():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.ShapeVar)

    ret_type = relay.TupleType(tvm.runtime.convert([tp1, tp1, tp1]))

    tf = relay.FuncType(
        tvm.runtime.convert([]), ret_type, tvm.runtime.convert([tp1]), tvm.runtime.convert([])
    )


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_func_with_invalid_relation():
    tp1 = relay.TypeVar("tp1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.ShapeVar)
    tp3 = relay.TypeVar("tp3", relay.TypeKind.Constraint)

    func = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Identity")
    tr = relay.TypeRelation(func, tvm.runtime.convert([tp2, tp3]), 1, None)

    tf = relay.FuncType(
        tvm.runtime.convert([tp1]),
        tp1,
        tvm.runtime.convert([tp1, tp2, tp3]),
        tvm.runtime.convert([tr]),
    )


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_tuple_with_invalid_func():
    tensor_type = relay.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")

    tp1 = relay.TypeVar("tp1", relay.TypeKind.ShapeVar)
    tf = relay.FuncType(
        tvm.runtime.convert([]), tp1, tvm.runtime.convert([tp1]), tvm.runtime.convert([])
    )

    tup_ty = relay.TupleType(tvm.runtime.convert([tensor_type, tf]))