from drjit.ast import TreeSet

def test01_tset():
    assert TreeSet(['a', 'a', 'b', 'b.c', 'ab', 'a.b.c']) == TreeSet(['a', 'b', 'ab'])
    assert TreeSet() | TreeSet() == TreeSet()
    assert TreeSet(['a']) | TreeSet(['b', 'a.b']) == TreeSet(['a', 'b'])
    assert TreeSet(['b', 'a.b', 'ab']) | TreeSet(['a']) == TreeSet(['a', 'b', 'ab'])
    assert TreeSet(['a']) | TreeSet(['b', 'a.b', 'ab']) == TreeSet(['a', 'b', 'ab'])
    assert TreeSet(['b', 'a.b', 'ab']) & TreeSet(['a']) == TreeSet(['a.b'])
    assert TreeSet(['a']) & TreeSet(['b', 'a.b', 'ab']) == TreeSet(['a.b'])

