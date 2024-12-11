# Copyright 2019 The Lifetime Value Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for lifetime value package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='lifetime_value',
    version='0.1',
    description='Package for lifetime value',
    author='Google LLC',
    author_email='no-reply@google.com',
    url='https://github.com/google/lifetime_value',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.11.1',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'tensorflow-probability',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
