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
    version='0.0.6',
    description="Generate Mjsynth text data",
    packages=['text_renderer'],
    package_data={'text_renderer': [
        'data/font/*',
        'data/font/vn/*',
        'data/fill/*'
    ]},
    include_package_data=True,
    install_requirements=requirements,
    license='MIT license',
)
