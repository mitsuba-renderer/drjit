import drjit as dr
import drjit.nn as nn
import pytest
import sys


def skip_if_coopvec_not_supported(t):
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.CUDA:
        if dr.detail.cuda_version() < (12, 8):
            pytest.skip("CUDA driver does not support cooperative vectors (Driver R570) or later is required")
    elif backend == dr.JitBackend.LLVM:
        if dr.detail.llvm_version() < (17, 0):
            pytest.skip("LLVM does not support cooperative vectors, 17.0 or later is required")


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test01_mapping(t):
    """MutableMapping surface: iteration, indexed reads/writes, update, bad keys, delitem."""
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16

    net = nn.Sequential(
        nn.Linear(2, 4, bias=True),
        nn.ReLU(),
        nn.Linear(-1, 3, bias=False),
    ).alloc(TensorXf16, 2)

    expected = ['layers.0.weights', 'layers.0.bias', 'layers.2.weights']
    assert list(net.keys()) == expected
    assert list(iter(net)) == expected
    assert len(net) == len(expected)
    assert 'layers.0.weights' in net
    assert 'bogus' not in net

    items = dict(net.items())
    assert set(items) == set(expected)
    values = list(net.values())
    assert len(values) == 3 and all(isinstance(v, TensorXf16) for v in values)
    for k in expected:
        assert items[k] is net[k]

    new_w = dr.full(TensorXf16, 3.0, (4, 2))
    net['layers.0.weights'] = new_w
    assert net.layers[0].weights is new_w
    assert dr.all(net['layers.0.weights'] == TensorXf16(3), axis=None)

    net.update({
        'layers.0.bias': dr.full(TensorXf16, 0.5, 4),
        'layers.2.weights': dr.full(TensorXf16, -1.0, (3, 4)),
    })
    assert dr.all(net['layers.0.bias'] == TensorXf16(0.5), axis=None)
    assert dr.all(net['layers.2.weights'] == TensorXf16(-1.0), axis=None)

    with pytest.raises(KeyError):
        net['bogus']
    with pytest.raises(KeyError):
        net['bogus'] = dr.zeros(TensorXf16, (4, 2))
    with pytest.raises(TypeError):
        del net['layers.0.weights']


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test02_setitem_dtype_cast(t):
    """Setting an fp32 value into an fp16 slot casts automatically."""
    mod = sys.modules[t.__module__]
    TensorXf = mod.TensorXf
    TensorXf16 = mod.TensorXf16

    net = nn.Sequential(nn.Linear(2, 4, bias=True)).alloc(TensorXf16, 2)

    net['layers.0.weights'] = dr.full(TensorXf, 2.5, (4, 2))
    got = net['layers.0.weights']
    assert isinstance(got, TensorXf16)
    assert dr.all(got == TensorXf16(2.5), axis=None)


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test03_prefix(t):
    """A prefix passed to Sequential namespaces every mapping key."""
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16

    net = nn.Sequential(nn.Linear(2, 4), prefix='mlp').alloc(TensorXf16, 2)
    assert net.prefix == 'mlp'
    assert set(net.keys()) == {'mlp.layers.0.weights', 'mlp.layers.0.bias'}
    assert 'mlp.layers.0.weights' in net


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test04_linear_errors(t):
    """Linear rejects unallocated calls and CoopVec calls on unpacked weights."""
    skip_if_coopvec_not_supported(t)
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16

    x = nn.CoopVec(dr.full(t, 1.0, 8), dr.full(t, 2.0, 8))

    uninit_msg = (
        "drjit.nn.Linear: uninitialized network. Call "
        "'net = net.alloc(<Tensor type>)' to initialize the weight "
        "storage first."
    )
    with pytest.raises(RuntimeError) as excinfo:
        nn.Linear(2, 4)(x)
    assert str(excinfo.value) == uninit_msg

    unpacked_msg = (
        "drjit.nn.Linear: cooperative-vector evaluation requires "
        "packed weights. Call 'drjit.nn.pack(net)' to transform "
        "the network into a cooperative-vector layout."
    )
    net = nn.Sequential(nn.Linear(2, 4)).alloc(TensorXf16, 2)
    with pytest.raises(RuntimeError) as excinfo:
        net(x)
    assert str(excinfo.value) == unpacked_msg


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test05_opt_roundtrip(t):
    """opt.update(net) then net.update(opt) preserves dtype + shapes."""
    from drjit.opt import Adam
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16

    net = nn.Sequential(
        nn.Linear(2, 4, bias=True),
        nn.Linear(-1, 3, bias=False),
    ).alloc(TensorXf16, 2)

    opt = Adam(lr=1e-3)
    opt.update(net)

    for k in list(opt.keys()):
        opt[k] = dr.full(type(opt[k]), 0.125, opt[k].shape)

    net.update(opt)
    for k, v in net.items():
        assert isinstance(v, TensorXf16)
        assert dr.all(v == TensorXf16(0.125), axis=None)


# ---------------------------------------------------------------------------
# Consistency: tensor mode vs cooperative-vector mode
# ---------------------------------------------------------------------------


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test06_tensor_vs_coopvec_parity(t):
    """Forward + input-gradient parity on a multi-layer Sequential."""
    skip_if_coopvec_not_supported(t)
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16
    ArrayXf16 = mod.ArrayXf16
    Float16 = t

    tensor_net = nn.Sequential(
        nn.Linear(2, 4, bias=True),
        nn.ReLU(),
        nn.Linear(-1, 3, bias=True),
    ).alloc(TensorXf16, 2, rng=dr.rng(seed=0xc0ffee))
    coop_net = nn.pack(dr.copy(tensor_net), layout='training')

    C, N = 2, 5
    data = [[0.2 + 0.05 * i + 0.03 * j for j in range(N)] for i in range(C)]

    x_tensor = TensorXf16(data)
    dr.enable_grad(x_tensor)
    y_tensor = tensor_net(x_tensor)

    x_rows = [Float16(data[i]) for i in range(C)]
    dr.enable_grad(x_rows)
    y_coop = ArrayXf16(coop_net(nn.CoopVec(*x_rows)))

    assert y_tensor.shape[0] == len(y_coop)
    for i in range(y_tensor.shape[0]):
        assert dr.allclose(Float16(y_tensor[i]), y_coop[i], atol=2e-2)

    dr.backward(dr.sum(y_tensor.array))
    dr.backward(dr.sum(dr.sum(y_coop)))
    for i, r in enumerate(x_rows):
        assert dr.allclose(Float16(x_tensor.grad[i]), r.grad, atol=2e-2)


# Factory / has-weights pairs for test07. Layers with trainable weights
# must go through ``nn.pack`` before accepting CoopVec inputs; stateless
# activations use the same module instance in both modes.
_LAYER_CASES = [
    pytest.param(lambda: nn.Linear(-1, 4, bias=True), True, id='linear'),
    pytest.param(lambda: nn.Linear(-1, 4, bias=False), True, id='linear_no_bias'),
    pytest.param(lambda: nn.ReLU(), False, id='relu'),
    pytest.param(lambda: nn.LeakyReLU(negative_slope=0.1), False, id='leaky_relu'),
    pytest.param(lambda: nn.Exp2(), False, id='exp2'),
    pytest.param(lambda: nn.Exp(), False, id='exp'),
    pytest.param(lambda: nn.Tanh(), False, id='tanh'),
    pytest.param(lambda: nn.ScaleAdd(scale=2.0, offset=0.5), False, id='scale_add'),
    pytest.param(lambda: nn.TriEncode(octaves=2, shift=0.3), False, id='tri_encode'),
    pytest.param(lambda: nn.SinEncode(octaves=2, shift=0.3), False, id='sin_encode'),
]


@pytest.mark.parametrize('layer_ctor,has_weights', _LAYER_CASES)
@pytest.test_arrays('jit,shape=(*),float16,diff')
def test07_layer_mode_consistency(t, layer_ctor, has_weights):
    """Each nn layer must produce matching outputs and input gradients
    in tensor mode and in cooperative-vector mode."""
    skip_if_coopvec_not_supported(t)
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16
    ArrayXf16 = mod.ArrayXf16
    Float16 = t

    C, N = 3, 4

    tensor_net = nn.Sequential(layer_ctor()).alloc(
        TensorXf16, C, rng=dr.rng(seed=0xc0ffee)
    )
    if has_weights:
        coop_net = nn.pack(dr.copy(tensor_net), layout='training')
    else:
        coop_net = tensor_net

    # Inputs stay inside [0, 1) (period of the encoders) while still
    # varying enough to exercise non-trivial gradients for activations.
    data = [[0.1 + 0.15 * i + 0.07 * j for j in range(N)] for i in range(C)]

    x_tensor = TensorXf16(data)
    dr.enable_grad(x_tensor)
    y_tensor = tensor_net(x_tensor)

    x_rows = [Float16(data[i]) for i in range(C)]
    dr.enable_grad(x_rows)
    y_coop = ArrayXf16(coop_net(nn.CoopVec(*x_rows)))

    assert y_tensor.shape[0] == len(y_coop)
    for i in range(y_tensor.shape[0]):
        assert dr.allclose(Float16(y_tensor[i]), y_coop[i], atol=2e-2)

    dr.backward(dr.sum(y_tensor.array))
    dr.backward(dr.sum(dr.sum(y_coop)))
    for i, r in enumerate(x_rows):
        assert dr.allclose(Float16(x_tensor.grad[i]), r.grad, atol=2e-2)


@pytest.test_arrays('jit,shape=(*),float32,diff')
@pytest.mark.parametrize("octaves", [1, 4])
@pytest.mark.parametrize("shift", [0.0, 0.3])
@pytest.mark.parametrize("encoding_cls", [nn.TriEncode, nn.SinEncode])
def test08_encode_mode_parity(t, octaves, shift, encoding_cls):
    """Encoder tensor-mode output must match the per-channel CoopVec output
    entry-by-entry (in fp32, for both encodings, with/without shift)."""
    skip_if_coopvec_not_supported(t)
    mod = sys.modules[t.__module__]
    TensorXf = mod.TensorXf
    Float32 = t

    enc = encoding_cls(octaves=octaves, shift=shift).alloc(TensorXf, 3)

    C, N = 3, 6
    x_tensor = dr.full(TensorXf, 0.37, (C, N))
    y_tensor = enc(x_tensor)
    assert y_tensor.shape == (2 * octaves * C, N)

    cv = nn.CoopVec(*(dr.full(Float32, 0.37, N) for _ in range(C)))
    y_cv = list(enc(cv))
    assert len(y_cv) == 2 * octaves * C

    tol = 1e-5 if encoding_cls is nn.SinEncode else 1e-6
    for k in range(2 * octaves * C):
        assert dr.allclose(Float32(y_tensor[k]), y_cv[k], atol=tol)


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test09_muon_cross_mode_consistency(t):
    """Muon drives the same loss trajectory in tensor mode and in coopvec mode.

    Both runs share the alloc seed and input data, so the initial weights
    and targets match. The per-step Muon update is computed in fp32;
    coopvec mode evaluates through hardware fp16 matvec whereas tensor
    mode uses row-major fp16 matmul. They should agree to within a
    generous fp16 tolerance.
    """
    skip_if_coopvec_not_supported(t)
    from drjit.opt import Muon
    mod = sys.modules[t.__module__]
    TensorXf16 = mod.TensorXf16
    Float16 = t
    Float32 = mod.Float32

    n_iter = 12

    init_net = nn.Sequential(
        nn.Linear(4, 32, bias=False),
        nn.Tanh(),
        nn.Linear(-1, 4, bias=False),
    ).alloc(TensorXf16, 4)
    rng = dr.rng()
    x = rng.random(TensorXf16, (4, 16))
    y_ref = rng.random(TensorXf16, (4, 16))

    def run(use_coopvec):
        net = dr.copy(init_net)
        muon = Muon(lr=0.05, momentum=0.9, ns_steps=5)
        muon.update(net)

        losses = []
        for _ in range(n_iter):
            net.update(muon)
            if use_coopvec:
                packed_net = nn.pack(net, layout='training')
                cv_in = nn.CoopVec(*[Float16(x[i]) for i in range(4)])
                y_cv = list(packed_net(cv_in))
                loss = Float32(0)
                for i in range(4):
                    diff = y_cv[i] - Float16(y_ref[i])
                    loss += dr.sum(Float32(diff * diff))
            else:
                diff = (net(x) - y_ref).array
                loss = Float32(dr.sum(Float32(diff * diff)))
            dr.backward(loss)
            muon.step()
            losses.append(float(loss[0]))
        return losses

    losses_tensor  = run(use_coopvec=False)
    losses_coopvec = run(use_coopvec=True)

    for lt, lc in zip(losses_tensor, losses_coopvec):
        assert abs(lt - lc) <= 0.01 * abs(lt), \
            f"trajectories diverge: tensor={lt:.4f} coopvec={lc:.4f}"


# ---------------------------------------------------------------------------
# Closed-form formula checks
# ---------------------------------------------------------------------------


@pytest.test_arrays('jit,shape=(*),float32,diff')
def test10_triencode_formula(t):
    """TriEncode output matches the closed-form formula exactly."""
    skip_if_coopvec_not_supported(t)
    mod = sys.modules[t.__module__]
    TensorXf = mod.TensorXf

    octaves, shift = 3, 0.15
    enc = nn.TriEncode(octaves=octaves, shift=shift).alloc(TensorXf, 1)

    x = TensorXf([[0.11, 0.22, 0.33]])
    out = enc(x)

    def sin_tri(a):
        s = a - 0.25
        return 1 - 4 * abs(s - round(s))
    def cos_tri(a):
        return 1 - 4 * abs(a - round(a))

    for i in range(octaves):
        for n, xv in enumerate([0.11, 0.22, 0.33]):
            a = (2 ** i) * xv + i * shift
            assert abs(float(out[2*i + 0, n].array[0]) - sin_tri(a)) < 1e-6
            assert abs(float(out[2*i + 1, n].array[0]) - cos_tri(a)) < 1e-6


@pytest.test_arrays('jit,shape=(*),float32,diff')
def test11_sinencode_formula(t):
    """SinEncode output matches the closed-form formula (shift in periods)."""
    skip_if_coopvec_not_supported(t)
    mod = sys.modules[t.__module__]
    TensorXf = mod.TensorXf

    octaves, shift = 3, 0.15
    enc = nn.SinEncode(octaves=octaves, shift=shift).alloc(TensorXf, 1)

    x = TensorXf([[0.11, 0.22]])
    out = enc(x)

    import math
    for i in range(octaves):
        for n, xv in enumerate([0.11, 0.22]):
            angle = 2 * math.pi * ((2 ** i) * xv + i * shift)
            assert abs(float(out[2*i + 0, n].array[0]) - math.sin(angle)) < 1e-5
            assert abs(float(out[2*i + 1, n].array[0]) - math.cos(angle)) < 1e-5
