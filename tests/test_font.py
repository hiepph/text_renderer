import context
from text_renderer.font import FontState, ColorState


fs = FontState()
cs = ColorState()


def test_font():
    sample = fs.get_sample()
    assert sample is not None
    assert sample['font'].endswith('ttf')


def test_color():
    sample = cs.get_sample(2)
    assert sample is not None
