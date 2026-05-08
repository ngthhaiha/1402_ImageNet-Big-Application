#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17']

nvcc_args = ['-std=c++17']

setup(
    name='channelnorm_cuda',
    ext_modules=[
        CUDAExtension('channelnorm_cuda', [
            'channelnorm_cuda.cc',
            'channelnorm_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
