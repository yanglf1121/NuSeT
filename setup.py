from setuptools import setup, find_packages


entry_points = {'console_scripts': ['nuset_gui=NuSeT:main']}

install_requires = \
    [
        "numpy", # newer numpy won't work with scikit-learn 0.13
        "scikit-image~=0.15.0", # newer scikit-image don't have the `min_size` kwarg for skimage.morphology.remove_small_holes
        "tensorflow==1.15", # tensorflow 1.15 required numpy>=1.16
        "Pillow",
        "tqdm"
    ]

classifiers = \
    [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Science/Research"
    ]

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name='nuset',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    entry_points=entry_points,
    url='https://github.com/yanglf1121/NuSeT',
    license='MIT License',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    install_requires=install_requires
)
