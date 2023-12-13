from setuptools import setup, find_packages

setup(
    name='mbridges',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.10.6',
    install_requires= ['numpy>=1.24.3', 'scipy>=1.10.1', 'opt-einsum>=3.3.0', 'h5py>=3.9'],
    extras_require = {
      'gpu_acceleration': ['cupy']
      }
    )

