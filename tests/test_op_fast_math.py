import numpy as np
import scipy
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime


def test_fastmath():
    def test_apply(relay_op, name, f_numpy, low, high, step, dtype="float32"):
        a_np = np.arange(low, high, step).astype(dtype)
        b_np = f_numpy(a_np)

        x = relay.var("x", shape=a_np.shape, dtype="float32")
        y = relay_op(x)
        func = relay.Function([x], y)
        mod = tvm.IRModule.from_expr(func)

        with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
            graph, lib, params = relay.build(mod, target="llvm", params=None)

        # Check that the op related to fast math have been convered to function in lib
        func_name = "fused_" + name
        assert lib.get_function(func_name)

        ctx = tvm.cpu(0)
        m = graph_runtime.create(graph, lib, ctx)
        # Set inputs
        m.set_input("x", tvm.nd.array(a_np, ctx))
        m.set_input(**params)
        # Execute
        m.run()
        # Get outputs
        tvm_output = m.get_output(0)