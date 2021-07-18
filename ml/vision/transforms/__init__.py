from ml import logging

try:
    import torchvision
    major, minor, patch = map(int, torchvision.__version__.split('.'))

    from torchvision.transforms import *
except Exception as e:
    logging.warn("torchvision unavailable, run `mamba install torchvision -c pytorch` to install")

from .transforms import (
    Resize, 
    ToCV
)

from .functional import *