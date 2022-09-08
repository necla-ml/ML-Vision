#!/usr/bin/env python

import os
import sys
import shutil
import distutils.command.clean
from pathlib import Path
from collections import OrderedDict

from setuptools import setup, find_namespace_packages
from pkg_resources import get_distribution, DistributionNotFound
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

from ml.shutil import run as sh
from ml import logging

named_arches = OrderedDict([
    ('Kepler+Tesla', '3.7'),
    ('Kepler', '3.5+PTX'),
    ('Maxwell+Tegra', '5.3'),
    ('Maxwell', '5.0;5.2+PTX'),
    ('Pascal', '6.0;6.1+PTX'),
    ('Volta', '7.0+PTX'),
    ('Turing', '7.5+PTX'),
    ('Ampere', '8.0;8.6+PTX'),
])

supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                    '7.0', '7.2', '7.5', '8.0', '8.6']

valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

# SM52 or SM_52, compute_52 
#   – Quadro M6000, 
#   - GeForce 900, GTX-970, GTX-980, 
#   - GTX Titan X
# SM61 or SM_61, compute_61 
#   – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030,
#   - Titan Xp, Tesla P40, Tesla P4, 
#   - Discrete GPU on the NVIDIA Drive PX2
# SM70 or SM_70, compute_70
#   - Titan V 
# SM75 or SM_75, compute_75 
#   – GTX/RTX Turing 
#   – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, 
#   - Titan RTX,

if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
    os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(valid_arch_strings)
    logging.warn(f'TORCH_CUDA_ARCH_LIST not set, build based on all valid arch: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    

cwd = Path(__file__).parent
pkg = sh('basename -s .git `git config --get remote.origin.url`').lower()
PKG = pkg.upper()

def write_version_py(path, major=None, minor=None, patch=None, suffix='', sha='Unknown'):
    if major is None or minor is None or patch is None:
        major, minor, patch = sh("git describe --abbrev=0 --tags")[1:].split('.')
        sha = sh("git rev-parse HEAD")
        logging.info(f"Build version {major}.{minor}.{patch}-{sha}")

    path = Path(path).resolve()
    pkg = path.name
    PKG = pkg.upper()
    version = f'{major}.{minor}.{patch}{suffix}'
    if os.getenv(f'{PKG}_BUILD_VERSION'):
        assert os.getenv(f'{PKG}_BUILD_NUMBER') is not None
        build_number = int(os.getenv(f'{PKG}_BUILD_NUMBER'))
        version = os.getenv(f'{PKG}_BUILD_VERSION')
        if build_number > 1:
            version += '.post' + str(build_number)
    elif sha != 'Unknown':
        version += '+' + sha[:7]

    import time
    content = f"""# GENERATED VERSION FILE
# TIME: {time.asctime()}
__version__ = {repr(version)}
git_version = {repr(sha)}

import torchvision
cuda = torchvision.version.cuda
"""

    with open(path / 'version.py', 'w') as f:
        f.write(content)
    
    return version


def dist_info(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


def ext_modules(pkg):
    pkg_dir = cwd / pkg
    extensions_dir = pkg_dir / 'csrc'

    main_files = sorted(map(str, extensions_dir.glob('*.cpp')))
    source_cpu = sorted(map(str, (extensions_dir / 'cpu').glob('*.cpp')))
    source_cuda = sorted(map(str, (extensions_dir / 'cuda').glob('*.cu')))

    sources = main_files + source_cpu
    extension = CppExtension
    if not sources:
        return []

    test_dir = cwd / 'tests'
    test_files = sorted(map(str, test_dir.glob('*.cpp')))
    tests = test_files

    define_macros = []
    extra_compile_args = {}
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args = {
            'cxx': ['-O3'],
            'nvcc': nvcc_flags,
        }

    if sys.platform == 'win32':
        define_macros += [('{pkg}_EXPORTS', None)]

    include_dirs = [str(extensions_dir)]
    tests_include_dirs = [str(test_dir)]
    ext_modules = [
        extension(
            f'{pkg}._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )]
    if tests:
        ext_modules.append(extension(
            f'{pkg}._C_tests',
            tests,
            include_dirs=tests_include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ))
    return ext_modules

# TODO
class Clean(distutils.command.clean.clean):
    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        print(f"removing {filename} to clean")
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    namespaces = ['ml']
    packages = find_namespace_packages(include=['ml.*'], exclude=('ml.csrc', 'ml.csrc.*'))
    version = write_version_py(pkg.replace('-', '/'))

    cmdclass = dict(
        build_ext=torch.utils.cpp_extension.BuildExtension,
        clean=Clean,
    )
    extensions = [ext for ext in ext_modules(pkg.split('-')[0])]
    name = sh('basename -s .git `git config --get remote.origin.url`').upper()
    logging.info(f"Building ml.vision with TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    setup(
        name=name,
        version=version,
        author='Farley Lai',
        url='https://github.com/necla-ml/ML-Vision',
        description=f"NECLA ML-Vision Library",
        long_description=readme(),
        keywords='machine learning, computer vision',
        license='BSD-3',
        classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: macOS/Ubuntu 16.04+',
            'Development Status :: 1 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        namespace_packages=namespaces,
        packages=namespaces + packages,
        install_requires=['ml'],
        ext_modules=extensions,
        cmdclass=cmdclass,
        entry_points=dict(
            console_scripts=[
                'mlv=ml.vision.cli.main:launch',
                'convert_dataset_yolo5=ml.vision.scripts:convert_dataset_yolo5',
                'make_dataset_yolo5=ml.vision.scripts:make_dataset_yolo5',
                'train_yolo5=ml.vision.scripts:train_yolo5',
                'deploy_yolo5=ml.vision.scripts:deploy_yolo5',
                'track_video=ml.vision.scripts:track_video',
            ]
        ),
        zip_safe=False)
