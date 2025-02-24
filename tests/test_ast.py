import drjit as dr
import pytest

@pytest.test_arrays('shape=(*), uint32, jit')
@dr.syntax
def test01_comprehensions(t):
    # Before PR #252, the j variable would incorrectly be considered part of the
    # loop state, breaking compilation. However, variables in comprehensions
    # are isolated and hence the variable 'j' isn't set until we're inside the
    # loop.

    n = 3
    [j for j in range(n)] # List comprehension
    {j: 'value' for j in range(n)} # Dict comprehension
    {j for j in range(n)} # Set comprehension
    (j for j in range(n) if j > 2) # Generator expression

    i = dr.zeros(t, 1)
    result = dr.zeros(t)
    while i < 2:
        for j in range(n):
            result += i * j
        i += 1

    assert result[0] == 3
