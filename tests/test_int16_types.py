import drjit as dr
import pytest
import sys


# Test basic UInt16 and Int16 type existence and construction
@pytest.test_packages()
def test01_uint16_int16_basic(p):
    """Test that UInt16 and Int16 types exist and can be constructed"""
    # Check types exist in all modules
    assert hasattr(p, 'UInt16'), f"{p.__name__} should have UInt16"
    assert hasattr(p, 'Int16'), f"{p.__name__} should have Int16"

    UInt16 = p.UInt16
    Int16 = p.Int16

    # Basic construction
    u = UInt16(42)
    i = Int16(-42)

    assert u[0] == 42
    assert i[0] == -42

    # Check type
    assert dr.type_v(UInt16) == dr.VarType.UInt16
    assert dr.type_v(Int16) == dr.VarType.Int16

    # Check signed/unsigned traits
    assert dr.is_unsigned_v(UInt16)
    assert not dr.is_signed_v(UInt16)
    assert dr.is_signed_v(Int16)
    assert not dr.is_unsigned_v(Int16)

    # Check they're integral
    assert dr.is_integral_v(UInt16)
    assert dr.is_integral_v(Int16)


# Test initialization from sequences
@pytest.test_arrays('type=uint16, shape=(*)', 'type=int16, shape=(*)')
def test02_init_from_sequence(t):
    """Test UInt16/Int16 initialization from Python sequences"""
    is_unsigned = 'UInt' in t.__name__

    # From list
    if is_unsigned:
        v = t([1, 2, 3, 4, 5])
        expected = [1, 2, 3, 4, 5]
    else:
        v = t([-2, -1, 0, 1, 2])
        expected = [-2, -1, 0, 1, 2]

    assert len(v) == 5
    for i in range(5):
        assert v[i] == expected[i]

    # From tuple
    if is_unsigned:
        v2 = t((10, 20, 30))
        assert v2[0] == 10 and v2[1] == 20 and v2[2] == 30
    else:
        v2 = t((-10, 0, 10))
        assert v2[0] == -10 and v2[1] == 0 and v2[2] == 10

    # From range
    v3 = t(range(10))
    assert len(v3) == 10
    for i in range(10):
        assert v3[i] == i


# Test casting between types
@pytest.test_arrays('type=uint16, shape=(*)', 'type=int16, shape=(*)')
def test03_cast_uint16_int16(t):
    """Test casting to/from UInt16 and Int16"""
    mod = sys.modules[t.__module__]
    is_uint16 = 'UInt16' in t.__name__

    # Test cast from/to other 16-bit type
    if is_uint16:
        # UInt16 -> Int16 -> UInt16
        v = t([100, 200, 300])
        v_int = mod.Int16(v)
        v_back = t(v_int)
        assert dr.all(v == v_back)
    else:
        # Int16 -> UInt16 -> Int16
        v = t([-100, 0, 100])
        v_uint = mod.UInt16(v)
        v_back = t(v_uint)
        assert dr.all(v == v_back)

    # Test widening cast from UInt8/Int8
    if hasattr(mod, 'UInt8'):
        UInt8 = mod.UInt8
        v8 = UInt8([10, 20, 30])
        v16 = t(v8)
        assert dr.all(v16 == [10, 20, 30])

    # Test narrowing cast to/from UInt32
    if hasattr(mod, 'UInt32') and is_uint16:
        UInt32 = mod.UInt32
        v32 = UInt32([100, 200, 300])
        v16 = t(v32)
        assert dr.all(v16 == [100, 200, 300])

        # Back to 32-bit
        v32_back = UInt32(v16)
        assert dr.all(v32_back == [100, 200, 300])

    # Test cast from Float32
    if hasattr(mod, 'Float'):
        Float = mod.Float
        vf = Float([10.5, 20.7, 30.1])
        v16 = t(vf)
        if is_uint16:
            assert dr.all(v16 == [10, 20, 30])
        else:
            assert dr.all(v16 == [10, 20, 30])

        # Back to float
        vf_back = Float(v16)
        if is_uint16:
            assert dr.all(vf_back == [10, 20, 30])
        else:
            assert dr.all(vf_back == [10, 20, 30])


# Test arithmetic operations
@pytest.test_arrays('type=uint16, shape=(*)', 'type=int16, shape=(*)')
def test04_arithmetic_uint16_int16(t):
    """Test arithmetic operations on UInt16/Int16"""
    is_unsigned = 'UInt' in t.__name__

    if is_unsigned:
        a = t([10, 20, 30, 40])
        b = t([1, 2, 3, 4])

        # Addition
        c = a + b
        assert dr.all(c == [11, 22, 33, 44])

        # Subtraction
        c = a - b
        assert dr.all(c == [9, 18, 27, 36])

        # Multiplication
        c = a * b
        assert dr.all(c == [10, 40, 90, 160])

        # Division
        c = a // b
        assert dr.all(c == [10, 10, 10, 10])
    else:
        a = t([-10, -5, 5, 10])
        b = t([2, 2, 2, 2])

        # Addition
        c = a + b
        assert dr.all(c == [-8, -3, 7, 12])

        # Subtraction
        c = a - b
        assert dr.all(c == [-12, -7, 3, 8])

        # Multiplication
        c = a * b
        assert dr.all(c == [-20, -10, 10, 20])

        # Division
        c = a // b
        assert dr.all(c == [-5, -2, 2, 5])  # floor division

    # Bitwise operations
    if is_unsigned:
        x = t([0xFF, 0xF0, 0x0F, 0xAA])
        y = t([0x0F, 0x0F, 0xF0, 0x55])
    else:
        x = t([127, -128, 100, -100])
        y = t([1, 1, 50, 50])

    # AND
    z = x & y
    assert len(z) == 4

    # OR
    z = x | y
    assert len(z) == 4

    # XOR
    z = x ^ y
    assert len(z) == 4

    # NOT
    z = ~x
    assert len(z) == 4

    # Shifts (for unsigned)
    if is_unsigned:
        x = t([1, 2, 4, 8])
        z = x << t([1, 1, 1, 1])
        assert dr.all(z == [2, 4, 8, 16])

        z = x >> t([0, 1, 1, 1])
        assert dr.all(z == [1, 1, 2, 4])


# Test min/max and comparisons
@pytest.test_arrays('type=uint16, shape=(*)', 'type=int16, shape=(*)')
def test05_minmax_comparisons(t):
    """Test min/max operations and comparisons"""
    is_unsigned = 'UInt' in t.__name__

    if is_unsigned:
        a = t([10, 20, 30, 5])
        b = t([15, 10, 25, 40])
    else:
        a = t([-10, 20, -30, 5])
        b = t([15, -10, 25, -40])

    # Min/Max
    c = dr.minimum(a, b)
    d = dr.maximum(a, b)

    assert len(c) == 4
    assert len(d) == 4

    # Comparisons
    m1 = a < b
    m2 = a <= b
    m3 = a > b
    m4 = a >= b
    m5 = a == b
    m6 = a != b

    # All should be masks
    assert dr.is_mask_v(type(m1))
    assert len(m1) == 4


# Test JIT operations (gather, scatter, reductions)
@pytest.test_arrays('type=uint16, jit, shape=(*)', 'type=int16, jit, shape=(*)')
def test06_jit_operations(t):
    """Test JIT-specific operations for UInt16/Int16"""
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32

    is_unsigned = 'UInt' in t.__name__

    # Create test data
    if is_unsigned:
        source = t([10, 20, 30, 40, 50])
    else:
        source = t([-20, -10, 0, 10, 20])

    # Test gather
    index = UInt32([0, 2, 4, 1, 3])
    gathered = dr.gather(t, source, index)
    dr.eval(gathered)

    if is_unsigned:
        assert dr.all(gathered == [10, 30, 50, 20, 40])
    else:
        assert dr.all(gathered == [-20, 0, 20, -10, 10])

    # Test scatter
    target = dr.zeros(t, 5)
    if is_unsigned:
        values = t([100, 200, 300])
    else:
        values = t([-100, 0, 100])

    indices = UInt32([0, 2, 4])
    dr.scatter(target, values, indices)
    dr.eval(target)

    if is_unsigned:
        assert target[0] == 100
        assert target[2] == 200
        assert target[4] == 300
    else:
        assert target[0] == -100
        assert target[2] == 0
        assert target[4] == 100

    # Test reductions
    if is_unsigned:
        data = t([1, 2, 3, 4, 5])
        assert dr.sum(data) == 15
        assert dr.min(data) == 1
        assert dr.max(data) == 5
    else:
        data = t([-2, -1, 0, 1, 2])
        assert dr.sum(data) == 0
        assert dr.min(data) == -2
        assert dr.max(data) == 2


# Test edge cases and overflow
@pytest.test_arrays('type=uint16, shape=(*)', 'type=int16, shape=(*)')
def test07_edge_cases(t):
    """Test edge cases including overflow behavior"""
    is_unsigned = 'UInt' in t.__name__

    if is_unsigned:
        # Test max value
        max_val = t([65535])
        assert max_val[0] == 65535

        # Test overflow (wrapping)
        overflow = t([65535]) + t([1])
        # Note: overflow behavior is implementation-defined, but typically wraps
        assert overflow[0] == 0 or overflow[0] == 65536  # May wrap or saturate

        # Test zero
        zero = t([0])
        assert zero[0] == 0
    else:
        # Test max positive value
        max_pos = t([32767])
        assert max_pos[0] == 32767

        # Test min negative value
        min_neg = t([-32768])
        assert min_neg[0] == -32768

        # Test zero
        zero = t([0])
        assert zero[0] == 0

    # Test select operation
    mod = sys.modules[t.__module__]
    if hasattr(mod, 'Bool'):
        Mask = dr.mask_t(t)

        if is_unsigned:
            a = t([10, 20, 30, 40])
            b = t([100, 200, 300, 400])
        else:
            a = t([-10, -20, 30, 40])
            b = t([10, 20, -30, -40])

        mask = Mask([True, False, True, False])
        result = dr.select(mask, a, b)

        if is_unsigned:
            assert result[0] == 10
            assert result[1] == 200
            assert result[2] == 30
            assert result[3] == 400
        else:
            assert result[0] == -10
            assert result[1] == 20
            assert result[2] == 30
            assert result[3] == -40
