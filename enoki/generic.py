from enoki import Dynamic, Exception


def _check(*args):
    """ Validate the inputs of a generic array operation """
    sizes = [len(a) for a in args]
    size_result = max(sizes)

    for i, size in enumerate(sizes):
        if size != size_result and size != 1:
            size_str = ', '.join(sizes)
            raise Exception("Incompatible argument sizes: %s" % size_str)
        elif type(args[i]) is not type(args[0]):  # noqa
            raise Exception("Type mismatch!")

    size_init = size_result if args[0].Size == Dynamic else 0
    return (*sizes, args[0].empty_(size_init), size_result)


def _check_inplace(*args):
    """ Validate the inputs of a generic in-place array operation """
    sizes = [len(a) for a in args]
    size_result = max(sizes)

    for i, size in enumerate(sizes):
        if size != size_result and size != 1:
            size_str = ', '.join(sizes)
            raise Exception("Incompatible argument sizes: %s" % size_str)
        elif type(args[i]) is not type(args[0]):  # noqa
            raise Exception("Type mismatch!")

    if sizes[0] == 1 and size_result > 1:
        raise Exception("In-place operation involving a vectorial result and "
                        "a scalar destination operand!")

    return (*sizes, size_result)


def add(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] + a1[i if s0 > 1 else 0]
    return ar


def iadd(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    for i in range(sr):
        a0[i] += a1[i if s0 > 1 else 0]
    return a0


def sub(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] - a1[i if s0 > 1 else 0]
    return ar


def isub(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    for i in range(sr):
        a0[i] -= a1[i if s0 > 1 else 0]
    return a0


def mul(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] * a1[i if s0 > 1 else 0]
    return ar


def imul(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    for i in range(sr):
        a0[i] *= a1[i if s0 > 1 else 0]
    return a0
