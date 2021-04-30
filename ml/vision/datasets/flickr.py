import os
import sys
import re
import csv
import h5py
import pickle
import base64
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import torch as th
import torch.nn.functional as F
#from torchvision.datasets import Flickr30k

from ml import logging
from ml.utils import iou

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

"""
Hierarchy of HDF5 file:
{ 
  'pos_boxes': num_images x 2           # bbox start offset, bbox end offset
  'image_bb': num_boxes x 4             # x1, y1, x2, y2
  'spatial_features': num_boxes x 6     # scaled x1, y1, x2, y2, w, h
  'image_features': num_boxes x 2048
}
"""


def extract(split, infiles, path="data/flickr"):
    """
    Args:
        split: ['train', 'val', 'test']
        path: data/flickr
        infiles: list of tsvs
    """
    # Downloaded
    ids_file = {  # image ids by split
        "train": path / "train.txt",
        "val": path / "val.txt",
        "test": path / "test.txt",
    }
    # Generated
    indices_file = {  # image id => index to pos_boxes
        "train": path / "train_imgid2idx.pkl",
        "val": path / "val_imgid2idx.pkl",
        "test": path / "test_imgid2idx.pkl",
    }
    # Generated
    data_file = {  # output dataset
        "train": path / "train.hdf5",
        "val": path / "val.hdf5",
        "test": path / "test.hdf5",
    }
    known_num_boxes = {"train": 904930, "val": 29906, "test": 30034}

    image_ids = {int(line) for line in open(ids_file[split])}
    h = h5py.File(data_file[split], "w")
    if known_num_boxes[split] is None:
        # XXX never happen
        num_boxes = 0
        for infile in infiles:
            print("reading tsv...%s" % infile)
            with open(infile, "r+") as tsv_in_file:
                reader = csv.DictReader(
                    tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES
                )
                for item in reader:
                    item["num_boxes"] = int(item["num_boxes"])
                    image_id = int(item["image_id"])
                    if image_id in image_ids:
                        num_boxes += item["num_boxes"]
    else:
        num_boxes = known_num_boxes[split]

    logger.info(f"{split} num_boxes={num_boxes}")
    feature_length = 2048
    img_features = h.create_dataset("image_features", (num_boxes, feature_length), "f")
    img_bb = h.create_dataset("image_bb", (num_boxes, 4), "f")
    spatial_img_features = h.create_dataset("spatial_features", (num_boxes, 6), "f")
    pos_boxes = h.create_dataset("pos_boxes", (len(image_ids), 2), dtype="int32")

    indices = {}  # image id => offset
    counter = 0
    num_boxes = 0
    min_fixed_boxes = 10
    max_fixed_boxes = 100
    FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
    for infile in infiles:  # tsvs
        unknown_ids = []
        logger.info(f"processing {infile}...")
        with open(infile, "r+") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
            for item in reader:
                item["num_boxes"] = int(item["num_boxes"])
                item["boxes"] = bytes(item["boxes"], "utf")
                item["features"] = bytes(item["features"], "utf")
                image_id = int(item["image_id"])
                image_w = float(item["image_w"])
                image_h = float(item["image_h"])
                bboxes = np.frombuffer(
                    base64.decodestring(item["boxes"]), dtype=np.float32
                ).reshape((item["num_boxes"], -1))

                # bbox: x1, y1, x2, y2
                box_width = bboxes[:, 2] - bboxes[:, 0]
                box_height = bboxes[:, 3] - bboxes[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bboxes[:, 0] / image_w
                scaled_y = bboxes[:, 1] / image_h

                box_width = box_width[..., np.newaxis]
                box_height = box_height[..., np.newaxis]
                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]
                spatial_features = np.concatenate(
                    (
                        scaled_x,
                        scaled_y,
                        scaled_x + scaled_width,
                        scaled_y + scaled_height,
                        scaled_width,
                        scaled_height,
                    ),
                    axis=1,
                )

                if image_id in image_ids:
                    image_ids.remove(image_id)
                    indices[image_id] = counter  # index to pos_boxes
                    pos_boxes[counter, :] = np.array(
                        [num_boxes, num_boxes + item["num_boxes"]]
                    )
                    img_bb[num_boxes : num_boxes + item["num_boxes"], :] = bboxes
                    spatial_img_features[
                        num_boxes : num_boxes + item["num_boxes"], :
                    ] = spatial_features
                    img_features[
                        num_boxes : num_boxes + item["num_boxes"], :
                    ] = np.frombuffer(
                        base64.decodestring(item["features"]), dtype=np.float32
                    ).reshape(
                        (item["num_boxes"], -1)
                    )
                    counter += 1
                    num_boxes += item["num_boxes"]
                else:
                    # out of split
                    unknown_ids.append(image_id)

        logger.info(f"{len(unknown_ids)} out of {split} split ids...")
        logger.info(f"{len(image_ids)} image_ids left...")

    if len(image_ids) != 0:
        logger.warn("Warning: %s_image_ids is not empty" % split)

    pickle.dump(indices, open(indices_file[split], "wb"))
    h.close()
    logger.info(f"Saved {split} features to {data_file[split]}")
    return h


def lastTokenIndex(arr, sub):
    """Return the index of the last token in the sublist

    Args:
        arr: list of tokens
        sub: list of phrase tokens
    """

    sublen = len(sub)
    first = sub[0]
    indx = -1
    while True:
        try:
            indx = arr.index(first, indx + 1)
        except ValueError:
            break
        if sub == arr[indx : indx + sublen]:
            return indx + sublen - 1

    return -1


def detectGT(src_bboxes, dst_bboxes):
    """Find matched ROI indices By IoU >= 0.5.
    Args:
        src_bboxes: entity GT bboxes
        dst_bboxes: all detected ROI bboxes
    """

    indices = set()
    for src_bbox in src_bboxes:
        for i, dst_bbox in enumerate(dst_bboxes):
            if iou(src_bbox, dst_bbox) >= 0.5:
                indices.add(i)

    return sorted(indices)


def _load_flickr30k(split, path, imgid2idx, offsets, rois): #, tokenize, tensorize):
    """Load entries of entity bboxes by ids.

    imgid2idx: dict {image_id -> offset to ROI features/bboxes}
    path: saved path to Flickr30K annotation dataset
    """

    path = Path(path)
    cache = path / f"{split}_entities.pt"
    if cache.exists():
        logger.info(f"Loading entities from {cache}")
        return th.load(cache)

    logger.info(f"Extracting entities from scratch...")
    from xml.etree.ElementTree import parse
    pattern_cap = r"\[[^ ]+ "
    pattern_no = r"\/EN\#(\d+)"
    pattern_phrase = r"\[(.*?)\]"

    grounding = OrderedDict()
    captions = OrderedDict()
    missing_entity_count = defaultdict(int)
    multibox_entity_count = 0
    entries = []
    from tqdm import tqdm
    for imgid, imgidx in tqdm(imgid2idx.items(), desc=f"Loading Flickr30K"):
        phrase_file = path / f"Sentences/{imgid}.txt"  # entity coreference chain
        anno_file = path / f"Annotations/{imgid}.xml"  # entity bboxes
        with open(phrase_file, "r", encoding="utf-8") as f:
            sents = [x.strip() for x in f]

        # Parse Annotation to retrieve GT bboxes for each entity id
        #   GT box => one or more entity ids
        #   entity id => one or more GT boxes
        root = parse(anno_file).getroot()
        obj_elems = root.findall("./object")
        start, end = offsets[imgidx]
        bboxes = rois[start:end]  # detected bboxes
        eGTboxes = defaultdict(list)  # entity_id: one or more GT bboxes
        for elem in obj_elems:
            # Exceptions: too many, scene or non-visual
            if elem.find("bndbox") == None or len(elem.find("bndbox")) == 0:
                continue

            left = int(elem.findtext("./bndbox/xmin"))  # -1
            top = int(elem.findtext("./bndbox/ymin"))  # -1
            right = int(elem.findtext("./bndbox/xmax"))  # -1
            bottom = int(elem.findtext("./bndbox/ymax"))  # -1
            assert 0 < left and 0 < top
            for name in elem.findall("name"):
                eId = int(name.text)
                assert 0 < eId
                if eId in eGTboxes:
                    multibox_entity_count += 1
                eGTboxes[eId].append([left, top, right, bottom])

        # Parse Sentence: captions and phrases w/ grounding
        #   entity id => one or more phrases
        grounding[imgid] = 0
        captions[imgid] = len(sents)
        for sent_id, sent in enumerate(sents):
            caption = re.sub(pattern_cap, "", sent).replace("]", "")
            entities = []           
            for i, entity in enumerate(re.findall(pattern_phrase, sent)):
                info, phrase = entity.split(" ", 1)
                eId = int(re.findall(pattern_no, info)[0])
                types = info.split("/")[2:]

                # grounded RoIs
                if not eId in eGTboxes:
                    assert eId >= 0
                    for t in types:
                        missing_entity_count[t] += 1
                    continue

                # find matched ROI indices with entity GT boxes
                detected = detectGT(eGTboxes[eId], bboxes)
                entities.append((eId, types, phrase, detected))
                if not detected:
                    logger.warn(f"No object detection of GT: [{imgid}][{i}:{eId}]{phrase}")

            if not entities:
                logger.warn(f"[{imgid}] no entity RoIs found: {sent}")
                continue

            grounding[imgid] += 1
            entries.append({
                    "imgid": imgid,
                    "imgidx": imgidx,
                    "caption": caption,                   
                    "entities": entities,
            })

            """
            entities = re.findall(pattern_phrase, sent)  # entity phrases
            sTokens = tokenize(caption)
            ent_indices = []
            roi_indices = []
            eTypes = defaultdict(list)
            for i, entity in enumerate(entities):
                info, phrase = entity.split(" ", 1)
                eId = int(re.findall(pattern_no, info)[0])
                eType = info.split("/")[2:]  # potentially multiple
                eTypes[eId] += eType
                pTokens = tokenize(phrase)
                eIdx = lastTokenIndex(sTokens, pTokens)
                assert 0 < eId
                assert 0 <= eIdx

                # XXX Only for grounding entities
                # skip scene or non-visual entity w/o bboxes
                if not eId in eGTboxes:
                    if eId >= 0:
                        missing_entity_count[eType[0]] = (
                            missing_entity_count.get(eType[0], 0) + 1
                        )
                    continue

                # find matched ROI indices with entity GT boxes
                roi_idx = detectGT(eGTboxes[eId], bboxes)
                roi_indices.append(roi_idx)  # list of corresponding ROI indices
                ent_indices.append(eIdx)

            if 0 == len(ent_indices):
                logger.warn(f"[{imgid}] no entity RoIs found: {sent}")
                continue

            """

            """
            grounding[imgid] += 1
            etype_stoi = {
                "pad": -1,
                "people": 0,
                "clothing": 1,
                "bodyparts": 2,
                "animals": 3,
                "vehicles": 4,
                "instruments": 5,
                "scene": 6,
                "other": 7,
                # "notvisual":8 ,
            }

            MAX_TYPE_NUM = 3
            PAD = etype_stoi["pad"]
            for i, eType in enumerate(eTypes):
                assert MAX_TYPE_NUM >= len(eType)                       # potentially multiple
                eTypes[i] = list(etype_stoi[etype] for etype in etype)  # to list of type ids
                eTypes[i] += [PAD] * (MAX_TYPE_NUM - len(eType))        # padding
            assert len(roi_indices) == len(ent_indices)
            token_ids, token_mask, token_seg = tensorize(sTokens)
            entries.append({
                    "imgid": imgid,
                    "imgidx": imgidx,
                    
                    "token_ids": token_ids,
                    "token_mask": token_mask,
                    "token_segment": token_mask,

                    "ent_indices": ent_indices,
                    "roi_indices": roi_indices,
                    "ent_types": eTypes,
            })
            """
            

    if 0 < len(missing_entity_count.keys()):
        grounded = th.tensor(list(grounding.values()))
        cap = th.tensor(list(captions.values()))
        incomplete = (grounded < cap).sum().item()
        none = (grounded == 0).sum().item()
        if none > 0:
            idx = (grounded == 0).nonzero().view(-1)
            imgids = list(grounding.keys())
            imgids = tuple(imgids[i] for i in idx)
            logger.warn(f"images w/o entity grounding: {imgids}")

        logger.warn(
            f"{incomplete}/{len(grounding)} with incomplete caption grounding, {none}/{incomplete} w/o grounding"
        )
        logger.warn(f"missing_entity_count: {', '.join(f'{k}={v}' for k, v in missing_entity_count.items())}")
        logger.warn(f"multibox_entity_count={multibox_entity_count}")

    th.save(entries, cache)
    return entries


class Flickr30kEntities(object):
    res = {
        "images": "images",  # image folder
        "captions": "results.tsv",  # captions
        "sentences": "Sentences",  # entity coreference chains
        "annotations": "Annotations",  # entity bounding boxes
        "features": {
            "cfg": "features/resnet101-faster-rcnn-vg-100-2048",
            "train": [
                "train_flickr30k_resnet101_faster_rcnn_genome.tsv.1",
                "train_flickr30k_resnet101_faster_rcnn_genome.tsv.2",
            ],
            "val": ["val_flickr30k_resnet101_faster_rcnn_genome.tsv.3"],
            "test": ["test_flickr30k_resnet101_faster_rcnn_genome.tsv.3"],
        },
    }

    EType2Id = dict(people=0, clothing=1, bodyparts=2, animals=3, vehicles=4, instruments=5, scene=6, other=7)
    ETypes   = list(EType2Id.keys())

    def __init__(
        self,
        split,
        tokenization,
        path="data/flickr",
        max_tokens=80,
        max_entities=16,
        max_rois=100,
        transform=None,
        target_transform=None,
    ):
        # XXX Use Entlities from BAN instead
        import h5py
        path = Path(path)

        # ROI features
        h5 = path / f"{split}.hdf5"
        imgid2idx = path / f"{split}_imgid2idx.pkl"
        if not h5.exists() or not imgid2idx.exists():
            logging.warning(f"{h5} or {imgid2idx} not exist, extracting features on the fly...")
            prefix = path / self.res["features"]["cfg"]
            tsvs = [prefix / tsv for tsv in self.res["features"][split]]
            logger.info(f"Extracting ROI features from {prefix}")
            extract(split, tsvs, path)

        logger.info(f"Loading image/RoI features from {h5}")
        self.imgid2idx = pickle.load(open(imgid2idx, "rb"))
        with h5py.File(h5, "r") as h5:
            self.offsets = th.from_numpy(np.array(h5.get("pos_boxes")))
            self.features = th.from_numpy(np.array(h5.get("image_features")))
            self.spatials = th.from_numpy(np.array(h5.get("spatial_features")))
            self.rois = th.from_numpy(np.array(h5.get("image_bb")))

        # Entities and ground truth bboxes
        #   pos_box start offset => annotation
        self.max_tokens = max_tokens
        self.max_entities = max_entities
        self.max_rois = max_rois
        self.tokenization = tokenization
        self.annotations = _load_flickr30k(
            split, path, self.imgid2idx, self.offsets, self.rois #, self.tokenize, self.tensorize
        )

        if tokenization in ['bert', 'wordpiece']:
            from ml.nlp import bert
            bert.setup()

    def tensorize(self, entry):
        if self.tokenization in ['bert', 'wordpiece']:
            from ml.nlp import bert
            caption = entry["caption"].lower()
            entities = entry["entities"]
            tokens = bert.tokenize(caption, plain=False)
            indices = -th.ones(self.max_entities, dtype=th.long)               # padded with -1 up to max_entities
            target = th.zeros(self.max_entities, self.max_rois) # padded up to (max_entities x max_rois)
            eTypes = th.zeros(self.max_entities, len(Flickr30kEntities.ETypes))
            for i, entity in enumerate(entities):
                eId, types, phrase, rois = entity
                toks = bert.tokenize(phrase, plain=False)[1:-1]
                index = lastTokenIndex(tokens, toks)
                assert index >= 0, f"Locate no phrase[{i}]={toks}"
                if index < self.max_tokens:
                    indices[i] = index
                    target[i][rois] = 1
                    #logger.info(f"phrase[{i}:{eId}]={toks}, index={index}, rois={rois}")
                else:
                    logger.warn(f"Truncated phrase: '{phrase}' for last token index {index} >= {self.max_tokens}")
                eTypes[i][list(Flickr30kEntities.EType2Id[t] for t in types)] = 1
            
            # BERT ids, mask and segment by truncating or padding input tokens up to max_tokens
            token_ids, token_seg, token_mask = bert.tensorize(tokens, max_tokens=self.max_tokens)
            return (token_ids, token_seg, token_mask), indices, target, eTypes
        else:
            # LSTM
            raise NotImplementedError('tokenization for LSTM is not implemented yet')

    def validate(self):
        rois_max = 0
        tokens_max =0
        entities_max = 0
        for i, entry in enumerate(self.annotations):
            features, spatials, tokens, indices, target = self[i]
            rois_max = max(rois_max, len(features))
            entities_max = max(entities_max, (indices <= 0).nonzero()[0].item())
            if self.tokenization in ['bert', 'wordpiece']:
                tokens_max = max(tokens_max, tokens[1].sum().item())

        assert rois_max <= self.max_rois
        assert tokens_max <= self.max_tokens
        assert entities_max <= self.max_entities
        logger.info(f"max (rois, tokens, entities) = ({rois_max}, {tokens_max}, {entities_max})")

    def __getitem__(self, index):
        """
        Args:
            index: by caption index

        Return:
            features
            spatials
            mask
            tokens
            
            indices: entity indices in target
            target (#entities x # RoIs): grounded entity RoIs
        """

        entry = self.annotations[index]
        imgidx = entry["imgidx"]
        start, end = self.offsets[imgidx]
        features = self.features[start:end, :]
        spatials = self.spatials[start:end, :]
        features = F.pad(features, (0, 0, 0, self.max_rois - len(features)))
        spatials = F.pad(spatials, (0, 0, 0, self.max_rois - len(spatials)))
        rois     = (end - start).item()
        mask     = th.tensor([1] * rois + [0] * (self.max_rois - rois))
        tokens, indices, target, types = self.tensorize(entry)
        # logger.info(f"{start}, {end}, {rois}, {mask.sum()}")
        return (features, spatials, mask, *tokens), (indices, target, types)

    def __len__(self):
        return len(self.annotations)
