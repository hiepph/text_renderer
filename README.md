# Text Renderer

Python library to generate synthetic text image data followed method by M. Jaderberg, et al. ([Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition](https://arxiv.org/abs/1406.2227)).
![method](https://www.robots.ox.ac.uk/~vgg/data/text/synthflow.png)


## Setup

```
git clone https://github.com/hiepph/text_renderer
cd text_renderer
python setup.py install
```


## Examples

+ Generate single image:

```python
import text_renderer
import cv2

# return numpy.ndarray image, and corresponding (random upper/lower) label
im, _label = text_renderer.gen('chào')
cv2.imwrite('chào.jpg', im)
```

![demo](./misc/demo/chào.jpg)


## References

+ Code is modified from [Max Jaderberg](https://bitbucket.org/jaderberg/text-renderer/src/master/)
