from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('vcurve_fns.pyx'),
      include_dirs = [numpy.get_include()])
