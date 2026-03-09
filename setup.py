from setuptools import setup, find_packages

setup(
    name='gvae-tme',
    version='0.1.0',
    description='Graph VAE for Tumor Microenvironment Analysis',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'torch-geometric>=2.3',
        'scanpy>=1.9',
        'anndata>=0.8',
        'scikit-learn>=1.2',
        'pandas>=1.5',
        'numpy>=1.24',
    ],
    python_requires='>=3.9',
)