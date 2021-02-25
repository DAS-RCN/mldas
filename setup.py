from setuptools import setup,find_packages
from glob import glob

with open('README.md', 'r') as f:
    longdesc = f.read()

setup(
    name="mldas",
    version='1.0.3',
    description="Machine learning analysis tools for Distributed Acoustic Sensing data.",
    long_description=longdesc,
    long_description_content_type='text/markdown',
    author="Vincent Dumont",
    author_email="vincentdumont11@gmail.com",
    maintainer="Vincent Dumont",
    maintainer_email="vincentdumont11@gmail.com",
    url="https://ml4science.gitlab.io/mldas",
    packages=['mldas'],
    project_urls={
        "Source Code": "https://gitlab.com/ml4science/mldas",
    },
    install_requires=["h5py","hdf5storage","matplotlib","mpi4py","numpy","pillow","pyyaml","scipy","torch","torchvision"],
    classifiers=[
        'Intended Audience :: Science/Research',
        "License :: Other/Proprietary License",
        'Natural Language :: English',
        "Operating System :: OS Independent",
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
    ],

)
