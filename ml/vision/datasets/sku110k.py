from collections import defaultdict
from pathlib import Path
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
from ml import logging

SKU110K_CLASSES = [ 'object' ]
SPLITS = dict(
    train='train.txt',
    val='val.txt',
    test='test.txt'
)

def toYOLO(root, splits=['val'], rewrites=None):
    '''Generate splits and labels in YOLOv5/COCO format (class x_center y_center width height) 
    with source format in (image_name, x1, y1, x2, y2, label, image_width, image_height).
    '''
    root = Path(root)
    images = root / 'images'
    labels = root / 'labels'
    labels.mkdir(parents=True, exist_ok=True)
    for split in splits:
        annotations = f"{root}/annotations/annotations_{split}.csv"
        with open(annotations) as af:
            with open(root / SPLITS[split], 'w') as sf:
                prev = None
                lf = None
                for filename, x1, y1, x2, y2, label, width, height in [line.split(',') for line in af.readlines()]:
                    x1, y1, x2, y2, width, height = tuple(map(float, [x1, y1, x2, y2, width, height]))
                    xc, yc, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1+1, y2-y1+1
                    path = labels / f"{filename[:-len('.jpg')]}.txt"
                    if prev != path:
                        if lf is not None:
                            lf.close()
                            logging.info(f'Converted to {prev} from {annotations}')
                        print(images / filename, file=sf)
                        lf = open(path, 'w')
                        prev = path
                    cls = rewrites[0] if rewrites and 0 in rewrites else 0
                    print(f"{cls} {min(1, xc/width):.6f} {min(1, yc/height):.6f} {min(1, w/width):.6f} {min(1, h/height):.6f}", file=lf)
                if lf is not None:
                    lf.close()
                    logging.info(f'Converted to {prev} from {annotations}')
                logging.info(f"Done {root / f'{split}.txt'}")

class SKU110K(VisionDataset):
    def __init__(self, root, split='val', transforms=None, transform=None, target_transform=None):
        super(SKU110K, self).__init__(self, root, transforms, transform, target_transform)
        self.split = split
        self.images = Path(f"{root}/images")
        with open(f"{root}/annotations/annotations_{split}.csv") as f:
            entries = defaultdict(list)
            for filename, x1, y1, x2, y2, label, width, height in [line.split(',') for line in f.readlines()]:
                x1, y1, x2, y2 = tuple(map(float, [x1, y1, x2, y2]))
                entries[filename].append((x1, y1, x2, y2, SKU110K_CLASSES.index(label)))
            self.entries = entries
            self.filenames = list(self.entries.keys())

    def __getitem__(self, index):
        """
        Args:
            index(int): Index

        Returns:
            dets(Tensor[K, 5]): a tensor of K bboxes in (x1, y1, x2, y2, cls)
        """
        filename = self.filenames[index]
        entry = self.entries[filename]
        img = Image.open(self.images / filename).convert('RGB')
        dets = torch.Tensor(entry)
        return img, dets

    def __len__(self):
        return len(self.entries)
