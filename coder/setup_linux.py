#!/usr/bin/env python3
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='coder',
    ext_modules=[
        CppExtension('coder', [
            'python.cpp',
            'BitIoStream.cpp',
            'ArithmeticCoder.cpp'
        ])],
    cmdclass={
        'build_ext': BuildExtension
    })
