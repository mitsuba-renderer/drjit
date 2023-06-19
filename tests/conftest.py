import drjit as dr
import types
import pytest

array_types = []

for o1 in dr.__dict__.values():
    if isinstance(o1, types.ModuleType):
        for o2 in o1.__dict__.values():
            if isinstance(o2, type) and issubclass(o2, dr.ArrayBase):
                array_types.append(o2)

def find_arrays(query):
    queries = query.replace(' ', '').split(',')
    result = []
    for o in array_types:
        matched = True
        meta = o.__meta__
        for q in queries:
            if not q in meta:
                matched = False
                break
        if matched:
            result.append(o)
    return result;


def test_arrays(*queries):
    """
    Helper function used to parameterize testcases over Dr.Jit array types

    Takes a list of query strings that add or remove (if prefixed with '-')
    types based on matches against the ``__meta__`` descriptor.

    For example, ``@pytest.test_arrays("is_vector", "-bool")`` parameterizes
    the subsequent test function over vector-style arrays that don't have a
    boolean dtype.

    The type argument of the subsequent testcase must be named "t"
    """
    pos, neg = [], []
    npos = 0
    for query in queries:
        if query[0] != '-':
            pos += find_arrays(query)
            npos += 1
        else:
            neg += find_arrays(query[1:])

    if npos == 0:
        pos = array_types

    found = [x for x in pos if x not in neg]
    ids = [a.__module__ + '.' + a.__name__ for a in found]
    if len(found) == 0:
        raise Exception('Query failed')

    def wrapped(func):
        return pytest.mark.parametrize('t', found, ids=ids)(func)

    return wrapped

def pytest_configure():
    pytest.test_arrays = test_arrays
