import drjit as dr
import types
import pytest
import re

array_types = []

for o1 in dr.__dict__.values():
    if isinstance(o1, types.ModuleType):
        for o2 in o1.__dict__.values():
            if isinstance(o2, type) and issubclass(o2, dr.ArrayBase):
                array_types.append(o2)


def test_arrays(query=""):
    """
    Helper function used to parameterize testcases over Dr.Jit array types

    Takes a list of query strings that add or remove (if prefixed with '-')
    types based on matches against the ``__meta__`` descriptor.

    For example, ``@pytest.test_arrays("is_vector", "-bool")`` parameterizes
    the subsequent test function over vector-style arrays that don't have a
    boolean dtype.

    The type argument of the subsequent testcase must be named "t"
    """
    result = set(array_types)

    query_list = re.split(r',\s*(?![^()]*\))', query)
    for query in query_list:
        if len(query) == 0:
            continue

        remove = False
        if query[0] == '-':
            query = query[1:]
            remove = True

        found = set(a for a in array_types if query in a.__meta__)
        if remove:
            result = result.difference(found)
        else:
            result = result.intersection(found)

    ids = [a.__module__ + '.' + a.__name__ for a in result]
    if len(result) == 0:
        raise Exception('Query failed')

    def wrapped(func):
        return pytest.mark.parametrize('t', result, ids=ids)(func)

    return wrapped

def pytest_configure():
    pytest.test_arrays = test_arrays
