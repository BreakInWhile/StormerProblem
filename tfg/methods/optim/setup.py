import Cython.Compiler.Options
from setuptools import Extension,setup
from Cython.Build import cythonize
import numpy

Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        "*", 
        ["*.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
]

setup(
    name='SV',
    ext_modules=cythonize(extensions,annotate=True),
    include_dirs=[numpy.get_include()]
)
