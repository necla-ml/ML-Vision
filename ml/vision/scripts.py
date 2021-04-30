import os
from pathlib import Path
from ml import argparse, logging, time, sys

def track_video():
    """
    Usage:
        # images: /zdata/projects/shared/datasets/kinetics400/frames-5fps/val/abseiling/GwlcmI36imo_000127_000137
        track_video /path/to/video.mp4 [--det-tag v3.1] [--render-all]
        track_video /path/to/image.jpg [--det-tag v3.1] [--render-all]
        track_video /path/to/images/ [--det-tag v3.1] [--render-all]

    Options:
        --det-chkpt-url s3://latest-sinet-checkpoints/detector/yolo/yolo5x-custom_rama_new-v3.1.pt \
        --det-chkpt yolov5x_custom \
        --det-classes 5 \
        --render-all \
        --reload
    """
    parser = argparse.ArgumentParser('Deploy a trained YOLO5 checkpoint on S3 by stripping out training states')
    parser.add_argument('path', help='Path to a trained checkpoint for deployment')
    parser.add_argument('-o', '--output', default='export', help='Path to output visualizations')

    parser.add_argument('-b', '--batch-size', default=24, type=int, help='Batch size to perform object detection inference')
    parser.add_argument('--fps', default=5, type=int, help='Frames per second in the video')
    parser.add_argument('--reload', action='store_true', help='Forece to reload checkpoint')
    parser.add_argument('--det-amp', action='store_true', help='Inference in AMP')
    parser.add_argument('--det-chkpt', default='yolov5x', choices=['yolov5x', 'yolov5x_custom'], help='Checkpoint name to save locally')
    parser.add_argument('--det-chkpt-url', help='S3 URL to download checkpoint')
    parser.add_argument('--det-classes', type=int, default=80, help='Number of classes to detect by the model')
    parser.add_argument('--det-tag', default='v3.1', help='Object detector code base git tag')
    parser.add_argument('--det-scales', default=640, type=int, choices=[608, 640, 672, 736], help='Size to rescale input for object detection')
    parser.add_argument('--det-cls-thres', default=0.4, type=float, help='Object class confidence threshold')
    parser.add_argument('--det-nms-thres', default=0.5, type=float, help='NMS IoU threshold')
    parser.add_argument('--det-pooling', default=1, type=int, help='Object feature pooling size for tracking')
    parser.add_argument('--trk-cls-person', nargs='+', default=[0], help='One or more person classes to track')
    parser.add_argument('--trk-max-iou-dist', default=0.8, type=float, help='max (1 - IoU) distance to track ')
    parser.add_argument('--trk-max-feat-dist', default=0.1395, type=float, help='max (1 - feature similarity) distance to track')
    parser.add_argument('--trk-gating-kf', default='iain', choices=['org', 'iain', False], help='KF gating for Deep Sort')
    parser.add_argument('--trk-gating-thrd', default=50, type=float, help='KF gating threshold')
    parser.add_argument('--trk-gating-alpha', default=0.2, type=float, help='KF gating parameter')
    parser.add_argument('--render-all', action='store_true', help='Render all objects or person only')
    cfg = parser.parse_args()
    print(cfg)

    from ml.vision.models import yolo5x, yolo5
    from ml.vision.models.tracking.dsort import DSTracker
    from ml.vision.ops import dets_select
    from torchvision.datasets.folder import default_loader as loader
    from torchvision.transforms import functional as TF
    from ml import av, cv, hub, logging, time
    import numpy as np
    import torch as th
    path = Path(cfg.path)
    fps = cfg.fps
    src = None
    if path.suffix in ['.mp4', '.avi']:
        # path to video
        src = av.open(cfg.path)
        v = src.decode(video=0)
        codec = src.streams[0].codec_context
        fps = round(codec.framerate)
        logging.info(f"Tracking video@{float(fps):.2f}fps in {path}")
    else:
        # path to image or a directory of subsampled images
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted([f for f in path.iterdir() if f.is_file()])
        def framer():
            for p in paths:
                if True:
                    # Follow feature extraction to load images in accimage.Image followed by ToTensor
                    img = loader(str(p))
                    img = cv.fromTorch(TF.to_tensor(img))
                    yield img
                else:
                    yield cv.imread(p)
        v = framer()
        logging.info(f"Tracking {len(paths)} frames@{cfg.fps}fps in {path}")
    
    dev = th.cuda.default_stream().device if th.cuda.is_available() else 'cpu'
    if cfg.det_chkpt_url:
        model = yolo5
        spec = hub.parse(cfg.det_chkpt_url)
        s3 = spec['scheme'] == 's3://' and spec or None
        # detector = model(chkpt=cfg.det_chkpt, tag=cfg.det_tag, pretrained=True, classes=cfg.det_classes, fuse=True, pooling=cfg.det_pooling, force_reload=cfg.reload, s3=s3).to(dev)
        # detector = model(chkpt='yolov5x-v2.0', s3=dict(bucket='eigen-pretrained', key='detection/yolo/yolov5x-v2.0.pt'), pretrained=True, fuse=True, pooling=cfg.det_pooling, force_reload=cfg.reload).to(dev)
        # detector = model(chkpt='yolov5x-v1.0', s3=dict(bucket='eigen-pretrained', key='detection/yolo/yolov5x-v1.0.pt'), tag='v1.0', pretrained=True, fuse=True, pooling=cfg.det_pooling, force_reload=cfg.reload).to(dev)
        # detector = model(chkpt='yolov5x-store-v1.0', s3=dict(bucket='eigen-pretrained', key='detection/yolo/yolov5x-store-v1.0.pt'), tag='v1.0', pretrained=True, fuse=True, pooling=cfg.det_pooling, force_reload=cfg.reload).to(dev)
        # detector = model(chkpt='yolov5x-retail81-v1.0', s3=dict(bucket='eigen-pretrained', key='detection/yolo/yolov5x-retail81-v1.0.pt'), tag='v1.0', pretrained=True, fuse=True, pooling=cfg.det_pooling, force_reload=cfg.reload).to(dev)
        detector = model(classes=cfg.det_classes,
                         pretrained=True, 
                         chkpt=cfg.det_chkpt, 
                         tag=cfg.det_tag, 
                         s3=s3, 
                         fuse=True, 
                         pooling=cfg.det_pooling, 
                         force_reload=cfg.reload)
    else:
        model = yolo5x
        detector = model(tag=cfg.det_tag, pretrained=True, classes=cfg.det_classes, fuse=True, pooling=cfg.det_pooling, force_reload=cfg.reload).to(dev)

    tracker = DSTracker(max_feat_dist=cfg.trk_max_feat_dist,
                        max_iou_dist=cfg.trk_max_iou_dist,
                        max_age=cfg.fps * 2,
                        n_init=3,
                        nn_budget=cfg.fps * 2,
                        gating_kf=cfg.trk_gating_kf,
                        gating_thrd=cfg.trk_gating_thrd,
                        gating_alpha=cfg.trk_gating_alpha)
    export = Path(f'{cfg.output}/{path.stem}-{cfg.det_chkpt}_{cfg.det_tag}_{cfg.det_scales}_{cfg.det_cls_thres}_{cfg.det_nms_thres}')
    export.mkdir(parents=True, exist_ok=True)
    assert export.exists()

    logging.info(f"Saving tracked video and frames to {export}")
    media = av.open(f"{export}/{path.stem}-tracking.mp4", 'w')
    stream = media.add_stream('h264', cfg.fps)
    stream.bit_rate = 2000000
    def track_frames(frames, start, step):
        # Track person only
        with th.cuda.amp.autocast(enabled=cfg.det_amp):
            dets, features = detector.detect(frames, size=cfg.det_scales, conf_thres=cfg.det_cls_thres, iou_thres=cfg.det_nms_thres)
        persons = dets_select(dets, cfg.trk_cls_person)
        objs = [dets_f[~persons_f].cpu() for dets_f, persons_f in zip(dets, persons)]
        ppls = [dets_f[persons_f].cpu() for dets_f, persons_f in zip(dets, persons)]
        ppl_feats = [feats_f[persons_f].cpu() for feats_f, persons_f in zip(features, persons)]
        for j, (objs_f, ppls_f, ppl_feats_f) in enumerate(zip(objs, ppls, ppl_feats), start):
            logging.info(f"[{start + (j - start) * step}] objs: {tuple(objs_f.shape)}, ppls: {tuple(ppls_f.shape)}, feats: {tuple(ppl_feats_f.shape)}")
            assert objs_f.shape[1] == 4 + 1 + 1
            assert ppls_f.shape[1] == 4 + 1 + 1
            assert len(ppls) == len(ppl_feats)
            # assert ppl_feats.shape[1] == 256 + 512 + 1024
            assert ppl_feats_f.shape[1] == 320 + 640 + 1280
            matches = tracker.update(ppls_f, ppl_feats_f.view(len(ppl_feats_f), np.prod(ppl_feats_f.shape[1:])))
            snapshot = tracker.snapshot()
            tracks = []
            for tid, info in snapshot:
                tracks.append([tid, info])
            logging.debug(f"matches[{start + (j - start) * step}]: {matches}")
            logging.debug(f"snapshot[{start + (j - start) * step}]: {snapshot}")

            # Render both dets and tracks side by side
            frame_det = frames[j - start]
            frame_trk = frame_det.copy()
            H, W, C = frame_det.shape
            if cfg.render_all:
                frame_det = cv.render(frame_det, th.cat([ppls_f, objs_f]), show=False)
            else:
                frame_det = cv.render(frame_det, ppls_f, show=False)
            frame_trk = cv.render(frame_trk, tracks, show=False)
            frame = np.zeros((H, 2 * W, C), dtype=np.uint8)
            frame[:, :W, :] = frame_det
            frame[:, W:, :] = frame_trk
            cv.save(frame, export / f"frame{start + (j - start) * step:03d}.jpg")
            #cv.save(frame_det, export / f"frame_det{j:03d}.jpg")
            #cv.save(frame_trk, export / f"frame_trk{j:03d}.jpg")
            if media is not None:
                frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                packets = stream.encode(frame)
                media.mux(packets)
                logging.debug(f'Encoded: {len(packets)} {packets}, {frame}')

    frames = []
    BS = bs = cfg.batch_size
    step = 1 if src is None else fps // cfg.fps
    t = time.time()
    for i, frame in enumerate(v):
        if isinstance(frame, av.VideoFrame):
            frame = cv.cvtColor(frame.to_rgb().to_ndarray(), cv.COLOR_RGB2BGR)
            assert frame.data.contiguous
        if i == 0:
            stream.height = frame.shape[0]
            stream.width = frame.shape[1] * 2
        if src is not None and i % step != 0:
            continue
        
        frames.append(frame)
        if len(frames) < BS:
            continue
        assert len(frames) == BS
        track_frames(frames, i - (BS - 1) * step, step)
        frames.clear()
        bs = BS

    if frames:
        track_frames(frames, i - (len(frames) - 1) * step, step)
    if media is not None:
        packets = stream.encode(None)
        if packets:
            media.mux(packets)
        media.close()
    if src is not None:
        src.close()
    elapse = time.time() - t
    logging.info(f"Done tracking {path.name} in {elapse:.3f}s at {(i + 1) / step / elapse:.2f}fps")

def deploy_yolo5():
    parser = argparse.ArgumentParser('Deploy a trained YOLO5 checkpoint on S3 by stripping out training states')
    parser.add_argument('chkpt', help='Path to a trained checkpoint for deployment')
    parser.add_argument('--url', help='s3://bucket/key/to/chkpt.pt')
    cfg = parser.parse_args()    

    from ml.vision.models.detection.yolo5 import GITHUB
    from ml import hub
    repo = hub.repo(GITHUB, force_reload=False)
    chkpt = cfg.chkpt
    if chkpt is None:
        chkpt = f"{repo}/weights/best.pt"
    
    sys.add_path(repo)
    from utils.utils import strip_optimizer
    before = os.path.getsize(chkpt)
    strip_optimizer(chkpt)
    after = os.path.getsize(chkpt)
    logging.info(f"Optimized {chkpt} optimized from {before / 1024**2:.2f}MB to {after / 1024**2:.2f}MB")
    if cfg.url.startswith('s3://'):
        logging.info(f"Uploading to {cfg.url}...")
        parts = cfg.url[len('s3://'):].split('/')
        bucket = parts[0]
        key = '/'.join(parts[1:])
        hub.upload_s3(chkpt, bucket, key)
    else:
        ValueError(f"Unsupported URL to upload: {cfg.url}")

def train_yolo5():
    """Train YOLOv5 over a dataset in YOLO5/COCO format.
    
    Usage:
        train_yolo5 ../../datasets/Retail81 --with-coco --names object --device 1 --batch-size 32
    
    References:
        python train.py --data coco.yaml --cfg yolov5x.yaml --weights yolov5x.pt --device 1 --batch-size 16
    """
    parser = argparse.ArgumentParser('Train YOLO5 oover a dataset in YOLO5/COCO format')
    parser.add_argument('path', help='Path to a dataset in YOLO/COCO format')
    parser.add_argument('--arch', choices=['yolov5l', 'yolov5x'], default='yolov5x', help='YOLOv5 model architecture to train')
    parser.add_argument('--tag', choices=['v1.0', 'v2.0', 'v3.0'], default='v3.0', help='YOLOv5 repo version tag')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--names', nargs='*', default=[], help='class names')
    parser.add_argument('--with-coco', action='store_true', help='Including COCO80 classes')
    args = parser.parse_args()
    logging.info(f"dataset={args.path}, arch={args.arch}, names={args.names}, with_coco={args.with_coco}")
    
    path = Path(args.path).resolve()
    data = path / 'data'
    data.mkdir(exist_ok=True)
    data /= f"{path.name.lower()}.yaml"
    models = path / "models"
    models.mkdir(exist_ok=True)
    cfg = f"{args.arch}.yaml"
    weights = f"{args.arch}.pt"
    names = args.names
    if args.with_coco:
        from ml.vision.datasets.coco import COCO80_CLASSES
        names = COCO80_CLASSES + names
    nc = len(names)
    assert nc > 0
    with open(data, 'w') as f:
        yaml = f"""
train: { path / 'train.txt' }
val: { path / 'val.txt' }
nc: {nc}
names: {names}
"""
        print(yaml, file=f)
        logging.info(yaml)

    from ml.vision.models.detection.yolo5 import github
    from ml.utils import Config
    from ml import hub
    repo = hub.repo(github(tag=tag), force_reload=False)
    config = Config().load(f"{repo}/models/{cfg}")
    config.nc = nc
    config.save(models / cfg)
    
    os.chdir(repo)
    cmd = f"python train.py --data {data.resolve()} --cfg {models / cfg} --weights {weights} --device {args.device} --batch-size {args.batch_size} --epochs {args.epochs}"
    logging.info(f"wd={os.getcwd()}")
    logging.info(cmd)
    r = os.system(cmd)
    if r:
        logging.error(f"Failed train YOLOv5 over {path} with res={r}")
    else:
        # v1.0
        # logging.info(f"Best trained checkpoiont at {repo}/weights/best.pt")
        # last v1.0 and v2.0+
        logging.info(f"Best trained checkpoiont at {repo}/runs/expX/weights/best.pt")

def convert_dataset_yolo5():
    '''Convert a supported dataset to YOLO5/COCO format.
    Usage:
        convert_dataset_yolo5 /mnt/local/datasets/coco --splits train val
        convert_dataset_yolo5 /mnt/local/datasets/SKU110K --splits train val --rewrites 0:80
        convert_dataset_yolo5 /mnt/local/datasets/WiderPerson --splits train val --rewrites 1:0 2:0 3:0 4:81 5:82
    '''
    parser = argparse.ArgumentParser('Convert supported dataset to YOLO5 format: [COCO, WiderPerson, SKU110K]')
    parser.add_argument('path', help='Path to supported dataset')
    parser.add_argument('--splits', nargs='+', default=['val'], help='Dataset split(s) to convert')
    parser.add_argument('--rewrites', nargs='+', default=None, help='Mapping to rewrite classes')
    cfg = parser.parse_args()
    path = Path(cfg.path)
    rewrites = None
    if cfg.rewrites:
        import re
        rewrites = dict(map(int, re.split(r'[:=]', rewrite)) for rewrite in cfg.rewrites)
    logging.info(f"dataset={path.name}, splits={cfg.splits}. rewrites={rewrites}, path={path}")
    from ml.vision import datasets
    if hasattr(datasets, path.name.lower()):
        getattr(datasets, path.name.lower()).toYOLO(path, splits=cfg.splits, rewrites=rewrites)
    else:
        raise ValueError(f"Unsupported dataset f'{dataset}'")

def make_dataset_yolo5():
    '''Compose a dataset in YOLO5/COCO format out of one or more supported datasets.
    Usage:
        make_dataset_yolo5 coco/ SKU110K/ -o Retail81 --splits val train
    '''
    parser = argparse.ArgumentParser("Compose a dataset in YOLO5 format from one or more supported datasets")
    parser.add_argument('sources', nargs='+', help='One or more paths to supported source datasets')
    parser.add_argument('-o', '--output', required=True, help='Composed output dataset path')
    parser.add_argument('--splits', nargs='+', default=['val'], help='Dataset split(s) to convert')
    cfg = parser.parse_args()

    dataset = Path(cfg.output)
    dataset.mkdir(parents=True, exist_ok=True)
    images = dataset / 'images'
    labels = dataset / 'labels'
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    logging.info(f"Composing dataset={dataset.name}, splits={cfg.splits}, path={dataset}")
    
    splits = dict(
        train=[],
        val=[],
        test=[],
    )
    logging.info(f"Collecting source dataset splits")
    from ml.vision import datasets
    for src in cfg.sources:
        src = Path(src)
        if hasattr(datasets, src.name.lower()):
            ds = getattr(datasets, src.name.lower())
        else:
            raise ValueError(f"Unsupported source dataset f'{src}'")
        for split in cfg.splits:
            splits[split].append(str(src / ds.SPLITS[split]))
    
    for split, split_files in splits.items():
        if not split_files:
            continue
        t = time.time()
        files = '\n'.join(split_files)
        logging.info(f"Working on {split} split from \n{files}")
        paths = []
        for file in split_files:
            entries = open(file).read().splitlines()
            paths.extend(entries)
            logging.info(f"Included {len(entries)} entries from {file}")
        with open(dataset / f"{split}.txt", 'w') as sf:
            for path in paths:
                path = Path(path)
                img_path = images / path.name
                label_path = labels / f"{path.stem}.txt"
                if img_path.is_symlink():
                    img_path.unlink()
                    logging.warning(f"Removed existing {img_path}")
                if label_path.is_symlink():
                    label_path.unlink()
                    logging.warning(f"Removed existing {label_path}")
                img_path.symlink_to(path)
                print(img_path, '->', f"{path}")
                parent = str(path.parent).replace('images', 'labels')
                label_path.symlink_to(f"{parent}/{path.stem}.txt")
                print(label_path, '->', f"{parent}/{path.stem}.txt")
                print(img_path.resolve(), file=sf)
        t = time.time() - t
        logging.info(f"Processed and saved {len(paths)} entries to {sf.name} in {t:.3}s")
