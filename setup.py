import fnmatch
import setuptools
from setuptools import find_packages, setup, Extension
from setuptools.command.build_py import build_py as build_py_orig


setup(name='gym_unemployment',
    version='1.11.0',
    install_requires=['fin_benefits','numpy','pandas','tqdm','seaborn','matplotlib','ipython'],
	packages=find_packages(),
	
    author="Antti Tanskanen",
    author_email="antti.tanskanen@gmail.com",
    description="Finnish earning-related social security as a Python module",
    keywords="life cycle model",
)
