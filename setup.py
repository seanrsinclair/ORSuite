#!usr/bin/env python

from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'or_suite'))
from or_suite.version import VERSION

setup(name='or-suite',
	version=VERSION,
	description='OR-Suite: A set of environments for developing reinforcement learning agents for OR problems.',
	author='Christopher Archer, Siddhartha Banerjee, Shashank Pathak, Carrie Rucker, Sean Sinclair, Christina Yu',
	author_email = 'srs429@cornell.edu',
	license='MIT',
	url='https://github.com/seanrsinclair/ORSuite',
	packages = find_packages(),
	zip_safe = False,
	python_requires = '>=3.6'
)