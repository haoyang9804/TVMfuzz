import numpy as np
import tvm
from tvm import relay
from tvm import te 

def test_dyn_upsampling_run():
    def verify_upsampling(dshape, scale_h, scale_w, layout, method, align_corners=False):

        if layout == "NCHW":
            (n, c, h, w) = dshape
            x_data = np.random.uniform(size=(n, c, h, w)).astype("float32")

        elif layout == "NHWC":
            (n, h, w, c) = dshape
            x_data = np.random.uniform(size=(n, h, w, c)).astype("float32")

        if method == "nearest_neighbor":
            ref_res = tvm.topi.testing.upsampling_python(x_data, (scale_h, scale_w), layout)
        else:
            ref_res = tvm.topi.testing.bilinear_resize_python(
                x_data, (int(round(h * scale_h)), int(round(w * scale_w))), layout
            )
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        scale_h_var = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w_var = relay.var("scale_h", relay.TensorType((), "float32"))

        z = relay.nn.upsampling(
            x, scale_h_var, scale_w_var, method=method, layout=layout, align_corners=align_corners
        )


# tests upsampling type inference with scale_h passed in as a constant and scale_w as a variable
@tvm.testing.uses_gpu
def test_dyn_upsampling_infer_type_const():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")

    data = relay.var("data", relay.TensorType((n, c, h, w), "int8"))
    scale_w = relay.Var("scale_w", relay.TensorType((), "float32"))

    z = relay.nn.upsampling(data, 2.0, scale_w)


@tvm.testing.uses_gpu
def test_dyn_upsampling3d_run():
    def verify_upsampling3d(
        dshape, scale_d, scale_h, scale_w, layout, method, coord_trans="half_pixel"
    ):

        if layout == "NCDHW":
            (n, c, d, h, w) = dshape
            x_data = np.random.uniform(size=(n, c, d, h, w)).astype("float32")

        elif layout == "NDHWC":
            (n, d, h, w, c) = dshape
            x_data = np.random.uniform(size=(n, d, h, w, c)).astype("float32")

        if method == "nearest_neighbor":
            ref_res = tvm.topi.testing.upsampling3d_python(
                x_data, (scale_d, scale_h, scale_w), layout
            )
        else:
            ref_res = tvm.topi.testing.trilinear_resize3d_python(
                x_data,
                (int(round(d * scale_d)), int(round(h * scale_h)), int(round(w * scale_w))),
                layout,
            )
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        scale_d_var = relay.var("scale_d", relay.TensorType((), "float32"))
        scale_h_var = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w_var = relay.var("scale_h", relay.TensorType((), "float32"))

        z = relay.nn.upsampling3d(
            x,
            scale_d_var,
            scale_h_var,
            scale_w_var,
            method=method,
            layout=layout,
            coordinate_transformation_mode=coord_trans,
        )


# tests upsampling type inference with scale_h passed in as a constant and scale_w as a variable
def test_dyn_upsampling3d_infer_type_const():
    n, c, d, h, w = (
        te.size_var("n"),
        te.size_var("c"),
        te.size_var("d"),
        te.size_var("h"),
        te.size_var("w"),
    )

    data = relay.var("data", relay.TensorType((n, c, d, h, w), "int8"))
    scale_d = relay.Var("scale_h", relay.TensorType((), "float32"))
    scale_w = relay.Var("scale_w", relay.TensorType((), "float32"))

    z = relay.nn.upsampling3d(data, scale_d, 2.0, scale_w, layout="NCDHW", method="trilinear")


@tvm.testing.uses_gpu
def test_dyn_pad():
    def verify_pad(dshape, pad_width, pad_val, dtype):
        x = relay.var("x", relay.TensorType(dshape, dtype))
        ndim = len(dshape)
        pad_width_var = relay.var("pad_width_var", relay.TensorType((ndim, 2), "int64"))
        pad_val_var = relay.var("pad_val_var", relay.TensorType((), dtype))
        y = relay.nn.pad(x, pad_width_var, pad_val_var)

    def verify_pad_default_fill(dshape, pad_width, dtype):
        x = relay.var("x", relay.TensorType(dshape, dtype))
        ndim = len(dshape)
        pad_width_var = relay.var("pad_width_var", relay.TensorType((ndim, 2), "int64"))
        y = relay.nn.pad(x, pad_width_var)

    verify_pad((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), 2.0, "int32")
    verify_pad((2, 7), ((1, 4), (2, 2)), 4.0, "float64")
    verify_pad_default_fill((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), "float64")
    verify_pad_default_fill((2, 7), ((1, 4), (2, 2)), "int32")


if __name__ == "__main__":
    test_dyn_pad()
    test_dyn_upsampling_infer_type_const()
    test_dyn_upsampling_run()
