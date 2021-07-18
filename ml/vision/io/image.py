from pathlib import Path
from torchvision import io
import torch as th

from ..transforms import functional as F

def load(path, *args):
    return io.read_image(str(path), *args)

def save(src, path, q=None):
    """Encode image to save in jpeg or png.
    Args:
        src (Tensor[CHW, dtype=uint8], PIL.Image, accimage.Image)

    NOTE:
        No longer support opencv
    """
    if not F.is_tensor(src):
        src = F.to_tensor(src)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(io, 'write_file'):
        io.write_file(src, path, q)
    else:
        if path.suffix.lower() == '.jpg':
            io.write_jpeg(src, str(path), q or 75)
        elif path.suffix.lower() == '.png':
            io.write_png(src, str(path), q or 6)
        else:
            raise ValueError(f"Unsupported image format: {path.suffix}")