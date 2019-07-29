from setuptools import setup


requirements = [
    'opencv-python',
    'Pillow',
    'scipy',
    'pygame',
    'numpy',
    'tqdm'
]

setup(
    name='text_renderer',
    version='0.0.2',
    description="Generate Mjsynth text data",
    packages=['text_renderer'],
    install_requirements=requirements,
    license='MIT license',
)
