# setup.py
from setuptools import setup, find_packages

setup(
    name="Scud",
    version="0.1.0",
    description="GPU-accelerated Mixed Linear Model GWAS Analysis Tool",
    author="Bo Zhang",
    author_email="bozhang@ibcas.ac.cn",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'cyvcf2>=0.30.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'tqdm>=4.62.0',
        'scipy>=1.7.0'
    ],
    entry_points={
        'console_scripts': [
            'scud=gpu_mlm:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords='gwas bioinformatics genomics gpu cupy',
)