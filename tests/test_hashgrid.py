import drjit as dr
import pytest
import sys

@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.test_arrays("jit,shape=(*),float32,diff")
def test01_hashgrid_tcnn(t, dimension):
    """
    Tests that the hashgrid implementation produces the same results and gradients
    as the one in tinycudann.
    """
    try:
        import tinycudann as tcnn
    except ImportError:
        pytest.skip("This test requires tinycudann to be installed.")

    try:
        import torch
    except ImportError:
        pytest.skip("This test requires PyTorch to be installed.")

    m = sys.modules[t.__module__]
    Float32 = m.Float32
    ArrayXf = m.ArrayXf
    PCG32 = m.PCG32

    device = t(1, 2).torch().device

    n = 16

    config = {
        "hashmap_size": 2**16,
        "n_levels": 16,
        "base_resolution": 16,
        "per_level_scale": 2,
        "n_features_per_level": 2,
    }

    hg = dr.nn.HashGridEncoding(
        Float32,
        dimension,
        **config,
        align_corners=False,
        torchngp_compat=True,
    )

    config = {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 16,
        "base_resolution": 16,
        "per_level_scale": 2,
        "interpolation": "Linear",
    }

    hg_ref = tcnn.Encoding(
        dimension,
        config,
        dtype=torch.float32,
    )
    data = hg_ref.params.data
    data = torch.linspace(0, 1_000, data.shape[0], device=device, dtype=torch.float32)
    hg_ref.params.data = data.cuda()

    for param in hg_ref.parameters():
        param.requires_grad_(True)

    hg.set_params(Float32(data.to(device=device)))
    dr.enable_grad(hg.data)

    sampler = PCG32(n)

    x = [sampler.next_float32() for _ in range(dimension)]
    dr.make_opaque(x)
    x_torch = torch.stack([xx.torch() for xx in x], dim=1).to(device="cuda")

    res = hg(x)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    ref = hg_ref(x_torch)
    end.record()
    torch.cuda.synchronize()

    ref = ref.to(device)
    res_torch = res.torch().permute(1, 0)
    assert torch.allclose(res_torch, ref)

    ## gradients

    res = ArrayXf(res)
    loss_res = dr.mean(dr.square(res - 1), axis=None)

    dr.backward(loss_res)

    loss_ref = torch.mean(torch.square(ref - 1), dim=None)
    loss_ref.backward()
    grad_res = dr.grad(hg.data).torch()
    grad_ref = hg_ref.params.grad.to(device=device)

    assert torch.allclose(grad_res, grad_ref)


@pytest.test_arrays("jit,shape=(*),float32,diff")
def test02_hashgrid_ref(t):
    """
    Tests that the hashgrid implementation produces the same results and gradients,
    as tinycudann, with saved reference data.
    """

    m = sys.modules[t.__module__]
    Float32 = m.Float32
    ArrayXf = m.ArrayXf
    PCG32 = m.PCG32

    n = 4

    config = {
        "hashmap_size": 2**16,
        "n_levels": 16,
        "base_resolution": 16,
        "per_level_scale": 2,
        "n_features_per_level": 2,
    }

    hg = dr.nn.HashGridEncoding(
        Float32,
        3,
        **config,
        align_corners=False,
        torchngp_compat=True,
    )

    data = hg.data
    data = dr.linspace(Float32, 0, 1_000, data.shape[0])
    hg.set_params(data)
    dr.enable_grad(hg.data)

    sampler = PCG32(n)

    x = [sampler.next_float32(), sampler.next_float32(), sampler.next_float32()]
    dr.make_opaque(x)

    res = hg(x)

    # Reference data
    ref = [
        [2.0091782, 3.3853433, 2.1349537, 1.5373236],
        [2.009702, 3.385867, 2.1354773, 1.5378475],
        [19.319187, 31.258121, 20.813356, 15.602333],
        [19.319712, 31.258648, 20.813877, 15.602858],
        [82.210266, 91.74675, 82.839745, 78.82015],
        [82.21078, 91.74728, 82.84027, 78.82066],
        [153.39827, 154.80888, 152.27963, 118.10999],
        [153.39879, 154.8094, 152.28017, 118.11051],
        [224.21854, 200.20016, 220.80148, 213.58574],
        [224.21907, 200.20071, 220.80202, 213.58624],
        [279.48984, 273.01324, 268.23297, 286.47266],
        [279.49036, 273.01376, 268.2335, 286.47318],
        [346.69785, 351.10532, 353.82407, 346.19147],
        [346.69836, 351.10583, 353.82462, 346.192],
        [402.95966, 424.1198, 425.00317, 430.22342],
        [402.96017, 424.1203, 425.00372, 430.22394],
        [472.5495, 481.44485, 485.95703, 479.4993],
        [472.55002, 481.44534, 485.95755, 479.49982],
        [549.6837, 573.11414, 549.44934, 563.9181],
        [549.6842, 573.1146, 549.4498, 563.9187],
        [630.8403, 627.67163, 601.86285, 611.5191],
        [630.84076, 627.67224, 601.8634, 611.51965],
        [676.8451, 693.41394, 693.47345, 706.95087],
        [676.84564, 693.4145, 693.474, 706.95135],
        [744.865, 763.62524, 750.0215, 768.94684],
        [744.8654, 763.6258, 750.02203, 768.9473],
        [838.08276, 834.42194, 829.39764, 818.9452],
        [838.08325, 834.4224, 829.3982, 818.94574],
        [899.3081, 894.0666, 890.1944, 895.054],
        [899.30865, 894.06714, 890.195, 895.05457],
        [960.4274, 959.1223, 964.1252, 960.06976],
        [960.428, 959.12286, 964.1257, 960.0703],
    ]

    assert dr.allclose(res, ref)


@pytest.test_arrays("jit,shape=(*),float32,diff")
def test03_permutohedral(t):
    """
    Tests that it is possible to run the permutohedral encodings.
    """
    import warnings
    with warnings.catch_warnings(record = True) as w:
        encoding = dr.nn.PermutoEncoding(
            t,
            dimension=2,
            n_levels=1,
            n_features_per_level=1,
            base_resolution=2,
            align_corners=True,
        )
        assert len(w) == 1
        assert all(issubclass(warning.category, RuntimeWarning) for warning in w)
        assert (
            "The number of features per level should be a multiple of 2, but it was 1."
            in str(w[0].message)
        )

    m = sys.modules[t.__module__]
    Float32 = m.Float32
    ArrayXf = m.ArrayXf

    encoding.set_params(dr.arange(Float32, 4))

    x = [Float32(0, 0, 1, 1/3), Float32(0, 1, 1, 2/3)]

    res = encoding(x)


    ref = ArrayXf([Float32(0, 2, 3, (0 + 2 + 3) / 3)])

    assert dr.allclose(res, ref)


@pytest.test_arrays("jit,shape=(*),float16,diff")
def test04_initialization(t):
    """
    Tests that the correct number of parameters are computed.
    """
    hg = dr.nn.HashGridEncoding(
        t,
        3,
        hashmap_size= 2**16,
        n_levels= 16,
        base_resolution= 16,
        per_level_scale= 2,
        n_features_per_level= 2,
        align_corners=False,
        torchngp_compat=True,
    )

    hg.n_params == 1908736
