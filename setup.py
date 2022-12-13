"""setup.py
"""
import os
from distutils.core import setup

cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, 'readme.md'), encoding='utf-8') as fh:
    long_description = fh.read()
    
setup(
    name='amspp',
    version='0.0.1',
    description='A multi-modal spiking neural network dataset and dataset loader for PyTorch SNN-like models.',
    author='Kunologist',
    author_email='jiyikun2002@gmail.com',
    url='https://github.com/Gennadiyev/amspp',
    packages=['amspp'],
    install_requires=['torch', 'snntorch', 'loguru', 'numpy'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)
