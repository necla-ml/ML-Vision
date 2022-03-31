import os
import tempfile

import pytest
from ml import logging
from ml.av import io

from .fixtures import img, vid

@pytest.mark.essential
def test_io_image(img):
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/image.jpg"
        io.save(img, path)
        size = os.path.getsize(path)
        logging.info(f"saved an image to {path} of {size} bytes")
        assert size > 0

        target = io.load(path)
        assert target.shape == img.shape
        assert target.dtype == img.dtype

@pytest.mark.essential
def test_io_video(vid):
    with tempfile.TemporaryDirectory() as tmp:
        # FIXME: fps must be 11+
        path = f"{tmp}/video.mp4"
        io.save(vid, path, fps=12, video_codec='h264')      
        size = os.path.getsize(path)
        logging.info(f"saved video to {path} of {size} bytes")
        assert size > 0

        video, audio, meta = io.load(path)
        assert video.shape == vid.shape
        assert video.dtype == vid.dtype