import random
import numpy as np

def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Replicate
        # img4, labels4 = replicate(img4, labels4)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=self.mosaic_border)  # border to remove

    return img4, labels4

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False):
    dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                  augment=augment,  # augment images
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=rect,  # rectangular training
                                  cache_images=cache,
                                  single_cls=opt.single_cls,
                                  stride=int(stride),
                                  pad=pad)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset

class DetectionDataset(VisionDataset):
    def __init__(self, path, img_size=640, batch_size=16, 
                    augment=False, hyp=None, rect=False, image_weights=False,
                    cache_images=False, single_cls=False, stride=32, pad=0.0):
        '''Dataset for detection with augmentation at batch level.

        Args:
            path(List[str], str): list of file(s) containing image paths or the parent directory of images
        '''
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file containing image paths
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                elif os.path.isdir(p):  # image directory
                    f += glob.iglob(p + os.sep + '*.*')
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)

        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index per image
        nb = bi[-1] + 1  # number of batches
        self.n = n  # number of images
        self.batch = bi  # batch index of image
        
        self.augment = augment
        self.hyp = hyp
        self.img_size = img_size
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        # Define labels
        '''
        img_files in [...images...]+
        label_files in [...labels...]+
        '''
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in self.img_files]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            l = self.labels[i]  # label
            if l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)
        assert nf > 0, 'No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        '''Cache dataset labels, check images and read shapes.

        /path/to/labels/[train | val | test]2017/imgid.txt: annotations in normalized (cls, x1, y1, x2, y2)
        cache(dict):
            path to image: List[array[(cls, x1, y1, x2, y2)*], shape]
        '''
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def load_image(self, index):
        '''Load and rescale 1 image.
        Returns: 
            img(CV2 BGR): loaded and resized image
            size0(Tuple(int, int)): original height and width
            resized(Tuple(int, int)): resized height and width
        '''
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

    def load_mosaic(self, index):
        # loads images in a mosaic
        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = load_image(self, index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

            # Replicate
            # img4, labels4 = replicate(img4, labels4)

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4, labels4 = random_affine(img4, labels4,
                                    degrees=self.hyp['degrees'],
                                    translate=self.hyp['translate'],
                                    scale=self.hyp['scale'],
                                    shear=self.hyp['shear'],
                                    border=self.mosaic_border)  # border to remove

        return img4, labels4