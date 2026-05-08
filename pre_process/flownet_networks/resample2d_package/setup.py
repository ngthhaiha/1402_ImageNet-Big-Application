#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17']

nvcc_args = ['-std=c++17']

setup(
    name='resample2d_cuda',
    ext_modules=[
        CUDAExtension('resample2d_cuda', [
            'resample2d_cuda.cc',
            'resample2d_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
