import os
from os import name
from pathlib import Path
from torchvision.datasets import coco
from ... import logging

# torchvision: 91
COCO91_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# MMDet, RFCN and YOLO as in COCO category ids
COCO80_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO80_TO_91 = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

COCO91_TO_80 = { id: i for i, id in enumerate(COCO80_TO_91) }

SPLITS = dict(
    train='train2017.txt',
    val='val2017.txt',
    test='test2017.txt'
)

BASE_DOWNLOAD_URL = lambda split: f'http://images.cocodataset.org/zips/{split}.zip'

def toYOLO(root, splits=['val2017'], rewrites=None, exact=False):
    """Generate labels and splits in YOLOv5 format (xc, yc, w, h) from official annotations (x1, y1, w, h).
    """
    rewrites = rewrites or COCO91_TO_80
    for split in splits:
        ds = CocoDetection(root, split)
        path = Path(root, SPLITS[split])
        labels = Path(root, 'labels', split)
        labels.mkdir(parents=True, exist_ok=True)
        logging.info(f"{split} at {path.resolve()}")
        with open(path, 'w') as sf:
            logging.info(f"### Processing {split} split to save to {path} ###")
            for i, (img, entries) in enumerate(ds):
                W, H = img.size
                img_id = ds.ids[i]
                img_path = (ds.images / f"{img_id:012d}.jpg").resolve()
                label = labels / f"{img_id:012d}.txt"
                logging.info(f"Processing {len(entries)} annotations of {img_path} ({W}x{H}) ###")
                with open(label, 'w') as lf: 
                    for j, entry in enumerate(entries):
                        c = int(entry['category_id'])
                        cls = rewrites[c] if rewrites else c
                        x1, y1, w, h = tuple(map(lambda v: float(v), entry['bbox']))
                        if exact:
                            x2, y2 = x1+w-1, y1+h-1
                            xc, yc, w, h = (x1+x2) / (2*W), (y1+y2) / (2*H), w / W, h / H
                        else:
                            xc, yc, w, h = (x1+w/2) / W, (y1+h/2) / H, w / W, h / H
                        values = [f"{max(min(v, 1), 0):.6f}" for v in (xc, yc, w, h)]
                        print(f"{cls} {' '.join(values)}", file=lf)
                    logging.info(f"Done saving labels to {label} ###")
                print(img_path, file=sf)
            logging.info(f"### Done processing {split} split at {path} ###")

def download(split='val2017', reload=False):
    """
    Downloads COCO split
    Returns:
        Path to downloaded split directory
    """
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile
    from ml import hub

    download_url = BASE_DOWNLOAD_URL(split)
    download_dir = Path(os.path.join(hub.get_dir(), 'COCO'))
    download_dir.mkdir(exist_ok=True, parents=True)
    split_dir = Path(os.path.join(download_dir, f'{split}'))
 
    if split_dir.exists() and any(split_dir.iterdir()) and not reload:
        # already exists
        logging.info(f'Skipping download of COCO: {split} as it already exists')
    else:
        # download 
        with urlopen(download_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(download_dir)
        logging.info(f'Downloaded COCO: {split}')
        
    return str(split_dir)

class CocoDetection(coco.CocoDetection):
    def __init__(self, root, split, transforms=None, transform=None, target_transform=None):
        """
        Args:
            root(path-like): path to downloaded COCO dataset with images/ and annotations/ ready
            transforms:
            transform:
            target_transform:
        Returns:
            img(PIL.Image): image in PIL RGB
            target(List[Dict]): 
                image_id:
                id:
                categrory_id:
                bbox[List[x1, y1, x2, y2]],
                segmentation(List[x,y]+),
                area(float):
                iscrowd(bool):
        """
        path = Path(root)
        images = path / 'images' / split
        annFile = path / f'annotations/instances_{split}.json'
        super(CocoDetection, self).__init__(str(images), str(annFile), transforms, transform, target_transform)
        self.images = images
        self.annFile = annFile
