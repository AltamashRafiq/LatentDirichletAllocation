from setuptools import setup
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


install_requires = [
    'numpy',
    'pandas',
    'cppimport',
    'gensim',
    'stop_words', 
    'numba', 
    'cython', 
    'tqdm'
    ]

setup(
    name='LDA-STA663',
    version='0.0.30',
    author="Altamash, Bernardo, and David",
    description='LDA implementation with Collapsed Gibbs Sampling following Griffiths & Steyvers (2004)',
    url="https://github.com/",
    long_description=long_description,
    license = "LICENSE.txt",
    ext_modules = cythonize("lda_src/cython_gibbs.pyx"),
    include_dirs=[np.get_include()],
    packages=["lda_src"],
    include_package_data=True,
    install_requires = install_requires,
    python_requires=">=3.6",
)
