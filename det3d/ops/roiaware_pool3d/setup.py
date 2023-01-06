import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(name,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=name,
        sources=sources,
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


setup(
    name='roiaware_pool3d',
    ext_modules=[
        make_cuda_ext(
            name='roiaware_pool3d_cuda',
            sources=[
                'src/roiaware_pool3d.cpp',
                'src/roiaware_pool3d_kernel.cu',
            ]),
    ],
    cmdclass={'build_ext': BuildExtension})
