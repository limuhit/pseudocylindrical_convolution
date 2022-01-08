#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
'''
Uncomment the compute_80 for A100 and compute_86 for 3090
'''		
cxx_args = ['-std=c++14', '-DOK']
nvcc_args = [
	'-D__CUDA_NO_HALF_OPERATORS__',
	'-gencode', 'arch=compute_61,code=sm_61',
	'-gencode', 'arch=compute_75,code=sm_75',
	#'-gencode', 'arch=compute_80,code=sm_80',
	#'-gencode', 'arch=compute_86,code=sm_86'
]

setup(
    name='PCONV',
    packages=['PCONV_operator'],
    ext_modules=[
        CUDAExtension('PCONV', [
			'./extension/main.cpp',
			'./extension/math_cuda.cu',
			'./extension/projects_cuda.cu',
			'./extension/dtow_cuda.cu',
			'./extension/context_reshape_cuda.cu',
			'./extension/entropy_gmm_cuda.cu',
			'./extension/mask_constrain_cuda.cu',
			'./extension/sphere_slice_cuda.cu',
			'./extension/sphere_uslice_cuda.cu',
			'./extension/entropy_gmm_table_cuda.cu',
			'./extension/entropy_context_cuda.cu',
			'./extension/entropy_ctx_pad_run2_cuda.cu',
			'./extension/d_extract_cuda_v2.cu',
			'./extension/d_input_cuda_v2.cu',
			'./extension/entropy_conv_cuda_v2.cu',
			'./extension/pseudo_context_cuda.cu',
			'./extension/pseudo_pad.cu',
			'./extension/pseudo_fill_cuda.cu',
			'./extension/pseudo_entropy_context_cuda.cu',
			'./extension/pseudo_entropy_pad_cuda.cu',
			'./extension/pseudo_quant_cuda.cu',
			'./extension/pseudo_dquant_cuda.cu',
			'./extension/string2class.cc',
			'./extension/entropy_add_cuda.cu',
        ],
        include_dirs=['./extension'], 
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}, 
        libraries=['cublas'])
    ],
    cmdclass={
        'build_ext': BuildExtension
})