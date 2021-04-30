from pathlib import Path
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
from ml import logging

WIDERPERSON_CLASSES = [
        '_bg',
        'pedestrians',
        'riders',
        'partially-visible persons',
        'ignore regions',
        'crowd',
]

SPLITS = dict(
    train='train_.txt',
    val='val_.txt',
    test='test_.txt'
)

def toYOLO(root, splits=['val'], rewrites=None):
    '''Generate splits and labels in YOLOv5/COCO format (class x_center y_center width height) with source format in (cls, x1, y1, x2, y2).
    Args:
        root(path-like): path to dataset
        splits(List[str]): split(s) to work on
        rewrites(dict): mapping from original classes to custom classes
    '''
    from ml import cv
    root = Path(root)
    images = root / 'images'
    labels = root / 'labels'
    assert (root / 'Images').exists()
    if not images.exists():
        images.symlink_to(root / 'Images', target_is_directory=True)
    labels.mkdir(parents=True, exist_ok=True)
    for split in splits:
        src = root / f'{split}.txt'
        dest = root/ SPLITS[split]
        with open(src) as sf:
            with open(dest, 'w') as df:
                for line in sf.read().splitlines():
                    path = images / f"{line}.jpg"
                    print(path, file=df)
                    srcA = root / 'Annotations' / f"{line}.jpg.txt"
                    if srcA.exists():
                        destL = labels / f"{line}.txt"
                        img = cv.Image.open(path)
                        width, height = cv.PIL_exif_size(img)
                        with open(srcA) as slf:
                            count = int(slf.readline())
                            with open(destL, 'w') as dlf:
                                lines = slf.read().splitlines()
                                assert count == len(lines)
                                for line in lines:
                                    cls, x1, y1, x2, y2 = list(map(int, line.split()))
                                    xc, yc, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1+1, y2-y1+1
                                    cls = rewrites[cls] if rewrites and cls in rewrites else cls
                                    print(f"{cls} {min(1, xc/width):.6f} {min(1, yc/height):.6f} {min(1, w/width):.6f} {min(1, h/height):.6f}", file=dlf)
                        logging.info(f'[{split}] Converted {srcA} to {destL} w.r.t. image size ({width}, {height})')
                    else:
                        logging.info(f'[{split}] {srcA} not exist')
        logging.info(f'Done {src} to {dest}')

def load(path):
    with open(path) as f:
        count = int(f.readline())
        dets = f.readlines()
        assert count == len(dets)
        dets = [(x1, y2, x2, y2, cls) for cls, x1, y1, x2, y2 in [tuple(map(float, det.split())) for det in dets]]
        return torch.Tensor(dets)

class WiderPerson(VisionDataset):
    def __init__(self, root, split='val', transform=None, target_transform=None,):
        with open(f"{root}/{split}.txt") as f:
            self.ids = list(sorted(map(lambda s: s.strip(), f.readlines())))
        self.images = Path(f"{root}/Images")
        self.annotations = Path(f"{root}/Annotations")

    def __getitem__(self, index):
        """
        Args:
            index(int): Index

        Returns:
            dets(Tensor[K, 5]): a tensor of K bboxes in (x1, y1, x2, y2, cls)
        """
        filename = f"{self.ids[index]}.jpg"
        img = Image.open(self.images / filename).convert('RGB')
        dets = load(self.annotations / f"{self.ids[index]}.jpg.txt")
        return img, dets

    def __len__(self):
        return len(self.ids)