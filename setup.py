import subprocess
import sys

def install_package(pack_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pack_name])

# Install tensorflow
try:
    import tensorflow as tf
    tfVersion = tf.__version__.split('.')
    # if not using tensorflow 1
    if tfVersion[0] != '1':
        install_package('tensorflow==1.15')

    # if version of tensorflow 1 is too old
    elif int(tfVersion[1]) < 13:
        install_package('tensorflow==1.15')
    
except ImportError:
    install_package('tensorflow==1.15')

# install numpy
try:
    import numpy as np 
except ImportError:
    install_package('numpy')

# install scikit-image
try:
    from PIL import Image
except ImportError:
    install_package('scikit-image')

# install tqdm
try:
    from tqdm import tqdm
except ImportError:
    install_package('tqdm')