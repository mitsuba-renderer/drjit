import pytest
import drjit as dr
import drjit.nn as nn
import sys

@pytest.test_arrays("float32, jit, shape=(*)")
def test01_simple_linear(t):

    m = sys.modules[t.__module__]

    rng = dr.rng(42)

    # Create simple linear layer
    layer = nn.Linear(-1, 8)
    layer = layer.alloc(m.TensorXf16, 4, rng=rng)

    # Pack with both methods
    packed_nn, _ = nn.pack(layer, layout="training")
    packed_diff = nn.diff_pack(layer, layout="training")

    print(f"{packed_nn=}")
    print(f"{packed_diff=}")
