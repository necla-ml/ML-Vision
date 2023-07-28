import warnings
from .boxes import *
from .utils import *

try:
    from .roi_align import roi_align, RoIAlign
    from .roi_pool import roi_pool, RoIPool
    from .pooler import MultiScaleFusionRoIAlign
except ImportError as e:
    warnings.warn(f'ml-vision built without custom extensions, make sure to build with extensions enabled if you need to use `roi_align`, `roi_pool` and other custom ops: {e}')