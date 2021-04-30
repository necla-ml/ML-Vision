import sys
import importlib
from ml import logging

CLI = ['imagenet']

def launch():
    logging.debug(f"{' '.join(sys.argv)}")
    assert len(sys.argv) > 1
    cmd = sys.argv[1]
    m = importlib.import_module(f".{cmd}", 'ml.vision.cli')
    argv = sys.argv[1:]
    m.launch(argv)
    return
