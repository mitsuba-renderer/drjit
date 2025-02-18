import drjit as dr
from drjit.opt import Adam, SGD, RMSProp, GradScaler
import pytest


@pytest.test_arrays("is_diff,float,shape=(*)")
def test01_basic(t):
    opt = Adam(1e-3)
    assert opt.learning_rate() == 1e-3
    opt.set_learning_rate(1e-4)
    assert opt.learning_rate() == 1e-4

    with pytest.raises(KeyError, match="nonexistent"):
        opt["nonexistent"]

    with pytest.raises(TypeError, match='parameter "x" is not differentiable'):
        opt["x"] = 1  # type: ignore

    with pytest.raises(RuntimeError, match='parameter "x" is empty'):
        opt["x"] = t()

    # Ensure that the optimizer makes a copy
    x = t(1)
    opt["x"] = x
    x += 1
    assert (x == 2) and (opt["x"] == 1)
    assert dr.grad_enabled(opt["x"])

    opt.set_learning_rate(x=1e-5)
    assert opt.learning_rate() == 1e-4
    assert opt.learning_rate("x") == 1e-5

    opt.update(x=t(4))
    opt.update({"y": t(5)})
    assert opt["x"] == 4 and opt["y"] == 5
    assert len(opt) == 2
    assert list(opt.keys()) == ["x", "y"]
    assert list(opt.values()) == [t(4), t(5)]
    assert list(opt.items()) == [("x", t(4)), ("y", t(5))]
    assert "x" in opt and "z" not in opt
    del opt["x"]
    assert "x" not in opt and len(opt) == 1

    assert (
        str(opt) == "Adam[\n"
        "  state = ['y'],\n"
        "  lr = {'default': 0.0001},\n"
        "  beta = (0.9, 0.999),\n"
        "  epsilon = 1e-08\n"
        "]"
    )


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test02_sgd(t):
    # Spot-check a run of the SGD optimizer against PyTorch
    #
    # Reference:
    #
    # import torch
    #
    # x = torch.Tensor([1.0])
    # x.requires_grad_()
    # opt = torch.optim.SGD(lr = .1, params=(x,))
    #
    # for _ in range(10):
    #     opt.zero_grad()
    #     loss = (x - 2)**2
    #     loss.backward()
    #     opt.step()
    #     print(f'{x[0].item()}, ', end='')

    x = t(1)
    opt = SGD(lr=0.1, params={"x": x})

    xv = []
    for _ in range(10):
        dr.backward((opt["x"] - 2) ** 2)
        opt.step()
        xv.append(opt["x"][0])

    ref = [
        1.2000000476837158,
        1.3600000143051147,
        1.4880000352859497,
        1.590399980545044,
        1.672320008277893,
        1.7378560304641724,
        1.7902848720550537,
        1.8322279453277588,
        1.865782380104065,
        1.8926259279251099,
    ]
    assert dr.allclose(xv, ref)


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test03_sgd_momentum(t):
    # Spot-check a run of the SGD optimizer with momentum against PyTorch
    #
    # Reference:
    #
    # import torch
    #
    # x = torch.Tensor([1.0])
    # x.requires_grad_()
    # opt = torch.optim.SGD(lr = .1, momentum=.9, params=(x,))
    #
    # for _ in range(10):
    #     opt.zero_grad()
    #     loss = (x - 2)**2
    #     loss.backward()
    #     opt.step()
    #     print(f'{x[0].item()}, ', end='')
    x = t(1)
    opt = SGD(lr=0.1, momentum=0.9, params={"x": x})

    xv = []
    for _ in range(10):
        dr.backward((opt["x"] - 2) ** 2)
        opt.step()
        xv.append(opt["x"][0])

    ref = [
        1.2000000476837158,
        1.540000081062317,
        1.9380000829696655,
        2.3085999488830566,
        2.5804200172424316,
        2.7089738845825195,
        2.682877540588379,
        2.522815465927124,
        2.2741963863372803,
        1.9955999851226807,
    ]
    assert dr.allclose(xv, ref)


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test04_sgd_momentum_nesterov(t):
    # Spot-check a run of the SGD optimizer with momentum
    # and Nesterov-style update against PyTorch
    #
    # Reference:
    #
    # import torch
    #
    # x = torch.Tensor([1.0])
    # x.requires_grad_()
    # opt = torch.optim.SGD(lr = .1, momentum=.9, nesterov=True, params=(x,))
    #
    # for _ in range(10):
    #     opt.zero_grad()
    #     loss = (x - 2)**2
    #     loss.backward()
    #     opt.step()
    #     print(f'{x[0].item()}, ', end='')

    x = t(1)
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True, params={"x": x})

    xv = []
    for _ in range(10):
        dr.backward((opt["x"] - 2) ** 2)
        opt.step()
        xv.append(opt["x"][0])

    ref = [
        1.3799999952316284,
        1.7775999307632446,
        2.108351945877075,
        2.3248229026794434,
        2.415717363357544,
        2.3980178833007812,
        2.305670738220215,
        2.178046703338623,
        2.0505480766296387,
        1.9486393928527832,
    ]

    print(f"{ref=}")
    print(f"{xv=}")

    assert dr.allclose(xv, ref)


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test05_rmsprop(t):
    # Spot-check a run of the RMSProp optimizer against PyTorch
    #
    # Reference:
    #
    # import torch
    #
    # x = torch.Tensor([1.0])
    # x.requires_grad_()
    # opt = torch.optim.RMSprop(lr = .02, params=(x,))
    #
    # for i in range(10):
    #     opt.zero_grad()
    #     loss = (x - 2)**2
    #     loss.backward()
    #     opt.step()
    #     print(f'{x[0].item()}, ', end='')

    x = t(1)
    opt = RMSProp(lr=0.02, params={"x": x})

    xv = []
    for _ in range(10):
        dr.backward((opt["x"] - 2) ** 2)
        opt.step()
        xv.append(opt["x"][0])

    ref = [
        1.1999999284744263,
        1.3253216743469238,
        1.4191335439682007,
        1.4943490028381348,
        1.5568580627441406,
        1.609941005706787,
        1.655657410621643,
        1.6954096555709839,
        1.730210781097412,
        1.7608258724212646,
    ]
    assert dr.allclose(xv, ref)


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test06_adam(t):
    # Spot-check a run of the Adam optimizer against PyTorch
    #
    # Reference:
    #
    # import torch
    #
    # x = torch.Tensor([1.0])
    # x.requires_grad_()
    # opt = torch.optim.Adam(lr = .5, params=(x,))
    #
    # for _ in range(10):
    #     opt.zero_grad()
    #     loss = (x - 2)**2
    #     loss.backward()
    #     opt.step()
    #     print(f'{x[0].item()}, ', end='')

    x = t(1)
    opt = Adam(lr=0.5, params={"x": x})

    xv = []
    for _ in range(10):
        dr.backward((opt["x"] - 2) ** 2)
        opt.step()
        xv.append(opt["x"][0])

    ref = [
        1.5,
        1.9660897254943848,
        2.335904598236084,
        2.542322874069214,
        2.585475206375122,
        2.5110909938812256,
        2.36381196975708,
        2.179222583770752,
        1.9882574081420898,
        1.8196619749069214,
    ]
    assert dr.allclose(xv, ref)


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test07_adam_incompat_shape(t):
    opt = Adam(lr=1)
    opt["x"] = t(1, 2, 3)
    opt["x"].grad = t(-1, -1, -1)
    opt.step()
    assert dr.allclose(opt.state['x'][-1], 1e-3)
    assert dr.allclose(opt.state['x'][-2], -1e-1)
    opt["x"] = t(1, 2, 3)
    assert dr.allclose(opt.state['x'][-1], 1e-3)
    assert dr.allclose(opt.state['x'][-2], -1e-1)
    opt["x"] = t(1, 2, 3, 4)
    assert dr.allclose(opt.state['x'][-1], 0)
    assert dr.allclose(opt.state['x'][-2], 0)


@pytest.test_arrays("is_diff,float,shape=(*),float32")
def test08_amp(t):
    t16 = dr.float16_array_t(t)
    opt = Adam(lr=1, params={'x': dr.ones(t, 10)})
    scaler = GradScaler()

    for it in range(20):
        loss = t((t16(opt["x"]) - 2) ** 2)
        dr.backward(scaler.scale(loss))
        scaler.step(opt)
    assert dr.allclose(opt["x"], 2.00245)
