# ML-Vision

Common computer vision NNs and ops:
- torchvision
- YOLOv5
- DETR
- & more

## Installation

```sh
conda install -c necla-ml ml-vision
```

## Usage
- Sample YOLO5 inference
```py
import random

import torch
from torchvision.io import write_jpeg
from torchvision.utils import draw_bounding_boxes

from ml.vision.datasets.coco import COCO80_CLASSES

TAG = 'v6.0'

def detector(tag):
    from ml.vision.models import yolo5x
    detector = yolo5x(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False, unload_after=True)
    detector.eval()
    return detector.to('cuda' if torch.cuda.is_available() else 'cpu')

def get_image():
    from PIL import Image 
    import requests
    import torchvision.transforms as T

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    transform = T.Compose([
        T.ToTensor()
    ])
    im = transform(im)
    return im

# get sample image from coco set
img = get_image() 
# normalized float to uint8 batch
img = (img * 255).to(torch.uint8).unsqueeze(0).repeat(10, 1, 1, 1)

# init detector
det = detector(TAG)
dets, feats = det.detect(img, batch_preprocess=True)

# get boxes: x1, y1, x2, y2
boxes = dets[0][:, :4]
classes = [COCO80_CLASSES[i] for i in dets[0][:, -1].int().tolist()]
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(boxes))]

# draw bounding boxes
img = draw_bounding_boxes(img[0], boxes=boxes, labels=classes, colors=COLORS, fill=True, width=3)
# write rendered image to jpeg file
write_jpeg(img, 'infered.jpg')
```

- Sample DETR inference
```py
import random
import torch as th
from ml.vision.models import detr

from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_jpeg

def get_image():
    from PIL import Image 
    import requests
    import torchvision.transforms as T

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    transform = T.Compose([
        T.ToTensor()
    ])
    im = transform(im)
    return im

# get sample image from coco set
img = get_image()

# init detector
detector = detr(
        pretrained=True,
        pooling=1,
        backbone='resnet50',
        deformable=False,
        resize=(800, 800),
        unload_after=True,
    )

detector.eval()
detector.to('cuda' if th.cuda.is_available() else 'cpu')

# model forward
dets, rec = detector.detect([img], cls_thres=0.9)

# get boxes: x1, y1, x2, y2
boxes = dets[0][:, :4]
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(boxes))]

# draw bounding boxes 
img = draw_bounding_boxes((img * 255).to(th.uint8), boxes=boxes, colors=COLORS)
# write rendered image to jpeg file
write_jpeg(img, 'infered.jpg')
```