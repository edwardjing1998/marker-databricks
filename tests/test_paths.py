import pytest
from shared.paths import prefix, roots, scoped_prefix, output_base, boolean

@pytest.mark.parametrize('name', ['q.pdf', 'q.PNG', 'q.jpg', 'q.jpeg'])
def test_mapping(name):
    assert output_base('source/book/chapter/' + name, 'source/', 'generated/') == 'generated/book/chapter/q'

def test_nested_filter_preserves_hierarchy():
    assert scoped_prefix('source/book/chapter/', 'source/') == 'source/book/chapter/'
    assert output_base('source/book/chapter/q.pdf', 'source/', 'generated/') == 'generated/book/chapter/q'

@pytest.mark.parametrize('value', ['', '/source/', '../x', 'source/../x', 'source//x'])
def test_bad_prefix(value):
    with pytest.raises(ValueError): prefix(value)

@pytest.mark.parametrize('a,b', [('source/', 'source/out/'), ('generated/', 'generated/'), ('source/', '_marker_jobs/output/')])
def test_overlap(a, b):
    with pytest.raises(ValueError): roots(a, b)

def test_scope_escape():
    with pytest.raises(ValueError): scoped_prefix('private/', 'source/')

def test_bool():
    assert boolean('true') and not boolean('FALSE')
    with pytest.raises(ValueError): boolean('yes')
