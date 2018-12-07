
import io
import os
import re

from setuptools import find_packages
from setuptools import setup

# Read version number from darts/__init__.py
with open('darts/__init__.py') as f:
    VERSION = re.search("__version__ = ['\"]([^'\"]+)['\"]", f.read()).group(1)


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="darts",
    version=VERSION,
    url="https://github.com/quark0/darts",
    license='Apache v2',

    author="",
    author_email="",

    description="https://github.com/quark0/darts",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'tqdm'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)