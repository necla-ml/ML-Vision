# ML-Vision

Common computer vision NNs and ops

## Installation

```sh
conda install -c necla-ml ml-vision
```

## Usage

- Sample YOLOX inference
```py
import torch as th
from ml.vision.models import yolox_x
from ml.vision.datasets.coco import COCO80_CLASSES

from torchvision.io import write_jpeg
import torchvision.transforms.functional as TF
from torchvision.utils import _generate_color_palette, draw_bounding_boxes

def get_image():
    from PIL import Image 
    import requests
    import torchvision.transforms as T

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    im = TF.pil_to_tensor(im)
    return im

# set device
dev = 'cuda' if th.cuda.is_available() else 'cpu'

# get sample image from coco set
img = get_image()

# init detector
detector = yolox_x(pretrained=True)
detector.eval()
detector.to(dev)

# model forward
dets = detector.detect([img], cls_thres=0.9)

# get boxes: x1, y1, x2, y2
boxes = dets[0][:, :4]
classes = [COCO80_CLASSES[i] for i in dets[0][:, -1].int().tolist()]
colors = _generate_color_palette(len(boxes))

# draw bounding boxes 
img = draw_bounding_boxes(img, boxes=boxes, labels=classes, colors=colors)
# write rendered image to jpeg file
write_jpeg(img, 'infered.jpg')
```

- Sample DETR inference
```py
import torch as th
from ml.vision.models import detr
from ml.vision.datasets.coco import COCO91_CLASSES

from torchvision.io import write_jpeg
import torchvision.transforms.functional as TF
from torchvision.utils import _generate_color_palette, draw_bounding_boxes

def get_image():
    from PIL import Image 
    import requests
    import torchvision.transforms as T

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    im = TF.pil_to_tensor(im)
    return im

# set device
dev = 'cuda' if th.cuda.is_available() else 'cpu'

# get sample image from coco set
img = get_image()

# init detector
detector = detr(
        pretrained=True,
        backbone='resnet50',
        deformable=False,
        resize=(800, 800),
        unload_after=True,
    )

detector.eval()
detector.to(dev)

# model forward
dets = detector.detect([img], cls_thres=0.9)

# get boxes: x1, y1, x2, y2
boxes = dets[0][:, :4]
classes = [COCO91_CLASSES[i] for i in dets[0][:, -1].int().tolist()]
colors = _generate_color_palette(len(boxes))

# draw bounding boxes 
img = draw_bounding_boxes(img, boxes=boxes, labels=classes, colors=colors)
# write rendered image to jpeg file
write_jpeg(img, 'infered.jpg')
```