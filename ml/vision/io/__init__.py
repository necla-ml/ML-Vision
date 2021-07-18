"""File I/O and encode/decode APIs.
"""

__all__ = [
    "write_video",
    "read_video",
    "read_video_timestamps",
    #"_read_video_from_file",
    #"_read_video_timestamps_from_file",
    #"_probe_video_from_file",
    #"_read_video_from_memory",
    #"_read_video_timestamps_from_memory",
    #"_probe_video_from_memory",
    #"_HAS_VIDEO_OPT",
    #"_read_video_clip_from_memory",
    #"_read_video_meta_data",
    "VideoMetaData",
    "Timebase",
#    "ImageReadMode",
    "decode_image",
#    "decode_jpeg",
#    "decode_png",
    "encode_jpeg",
    "encode_png",
#    "read_file",
    "read_image",
#    "write_file",
    "write_jpeg",
    "write_png",
    #"Video",

#    "save",
#    "load",
]

from ml import logging

try:
    import torchvision
    major, minor, patch = map(int, torchvision.__version__.split('.'))

    from torchvision.io import (
        decode_image,
        encode_jpeg,
        encode_png,
        read_image,
        write_jpeg,
        write_png,

        read_video,
        read_video_timestamps,
        write_video,

        Timebase,
        VideoMetaData,

        VideoReader,
    )

    try:
        from torchvision.io import (
            ImageReadMode,  
            decode_jpeg,
            decode_png,
        )
        __all__.extend([
            'ImageReadMode',
            'decode_jpeg',
            'decode_png',
        ])
    except Exception as e:
        logging.warn(f"{e}, install torchvision=0.9.0+")

    try:
        from torchvision.io import (
            read_file,
            write_file,
        )
        __all__.extend([
            'read_file',
            'write_file',
        ])
    except Exception as e:
        logging.warn(f"{e}, install torchvision=0.10.0+")
except Exception as e:
    logging.warn(f"{e}, install torchvision=0.8.1+")


from .image import *
from .video import *


__all__.extend([
    'save',
    'load',
])