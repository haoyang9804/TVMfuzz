import numpy as np

import tvm
from tvm import relay
from tvm.relay import transform


def test_simple_graph():
    # A module with two subgraphs
    mod = tvm.IRModule()

    x0 = relay.var("x0", shape=(8, 8))
    y0 = relay.var("y0", shape=(8, 8))
    z0 = x0 + y0
    z1 = x0 - y0
    z2 = relay.Tuple((z0, z1))
    f0 = relay.Function([x0, y0], z2)
    f0 = f0.with_attr("Compiler", "test_graph")
    g0 = relay.GlobalVar("g0")
    mod[g0] = f0

    x1 = relay.var("x1", shape=(8, 8))
    y1 = relay.var("y1", shape=(8, 8))
    z1 = x1 - y1
    f1 = relay.Function([x1, y1], z1)
    f1 = f1.with_attr("Compiler", "test_graph")
    g1 = relay.GlobalVar("g1")
    mod[g1] = f1

    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))
    z = relay.var("z", shape=(8, 8))
    c0 = relay.Call(g0, [x, y])
    c1 = relay.Call(g1, [relay.TupleGetItem(c0, 0), z])
    fm = relay.Function([x, y, z], c1)
    mod["main"] = fm

    x_data = np.random.rand(8, 8).astype("float32")
    y_data = np.random.rand(8, 8).astype("float32")
    z_data = np.random.rand(8, 8).astype("float32")
    data = get_calibration_data(mod, {"x": x_data, "y": y_data, "z": z_data})

    # Check the number and orders
    check_data_size(mod, data)


def test_mobilenet_dnnl():
    # if not tvm.get_global_func("relay.ext.dnnl", True):
    #     print("skip because DNNL codegen is not available")
    #     return

    dtype = "float32"
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32")

    mod = transform.AnnotateTarget(["dnnl"])(mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)