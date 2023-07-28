import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


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
    parser.add_argument('--reload', action='store_true', help='Force to reload checkpoint')
    parser.add_argument('--det-amp', action='store_true', help='Inference in AMP')
    parser.add_argument('--det-chkpt', default='yolov5x', choices=['yolov5x', 'yolov5x_custom', 'yolov5m'], help='Checkpoint name to save locally')
    parser.add_argument('--det-backend', default=None, choices=['trt'], help='Inference backend to use')
    parser.add_argument('--det-trt-fp16', action='store_true',  help='TRT FP16 enabled or not')
    parser.add_argument('--det-trt-int8', action='store_true',  help='TRT INT8 enabled or not')
    parser.add_argument('--det-chkpt-url', help='S3 URL to download checkpoint')
    parser.add_argument('--det-classes', type=int, default=80, help='Number of classes to detect by the model')
    parser.add_argument('--det-dataset', type=str, default='coco', choices=['coco', 'object365'], help='Name of the pretrained model dataset')
    parser.add_argument('--det-tag', default='v6.0', help='Object detector code base git tag')
    parser.add_argument('--det-resize', default=[720, 1280], type=int, help='Resize frame')
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
    from torchvision.transforms import functional as TF
    from torchvision.io import write_jpeg, read_image
    from ml import av, hub, logging, time
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
                    img = read_image(str(p))
                    yield img
                else:
                    yield cv.imread(p)
        v = framer()
        logging.info(f"Tracking {len(paths)} frames@{cfg.fps}fps in {path}")
    
    # setup model
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
    
    if cfg.det_backend in ['trt']:
        import math
        # XXX Deployment by batch size and minimal preprocessed shape
        amp = cfg.det_trt_fp16
        bs = cfg.batch_size
        scale = cfg.det_scales
        H, W = cfg.det_resize
        if W > H:
            spec = (3, 32 * math.ceil(H / W * scale / 32), scale)
        else:
            spec = (3, scale, 32 * math.ceil(W / H * scale / 32))
        logging.info(f"Deploying runtime={cfg.det_backend} with batch_size={bs}, spec={spec}, fp16={amp}")
        detector.deploy('yolov5x', batch_size=bs, spec=spec, fp16=amp, int8=cfg.det_trt_int8, backend=cfg.det_backend, reload=False)

    # setup tracker
    tracker = DSTracker(max_feat_dist=cfg.trk_max_feat_dist,
                        max_iou_dist=cfg.trk_max_iou_dist,
                        max_age=cfg.fps * 2,
                        n_init=3,
                        nn_budget=cfg.fps * 2,
                        gating_kf=cfg.trk_gating_kf,
                        gating_thrd=cfg.trk_gating_thrd,
                        gating_alpha=cfg.trk_gating_alpha)

    export_path = Path(f'{cfg.output}/{path.stem}-{cfg.det_chkpt}_{cfg.det_tag}_{cfg.det_scales}_{cfg.det_cls_thres}_{cfg.det_nms_thres}')
    # create frame path
    export_frame = export_path / 'rendered_frames'
    export_frame.mkdir(parents=True, exist_ok=True)
    assert export_frame.exists()

    TRACK_INFO = 'x1, y1, x2, y2, score, class, origin_x, origin_y, velocity_x, velocity_y'
    DETECTION_INFO = 'x1, y1, x2, y2, score, class'
    annot_dict = defaultdict(dict)

    # setup dataset classes and colors
    from ml.vision.utils import COLORS
    from ml.vision.datasets.coco import COCO80_CLASSES
    from ml.vision.datasets.object365 import OBJECT365_CLASSES
    DATASETS = {
        'coco': COCO80_CLASSES,
        'object365': OBJECT365_CLASSES
    }
    assert cfg.det_dataset in DATASETS
    DATASET = DATASETS[cfg.det_dataset] 
    COLOR = COLORS(len(DATASET))
    logging.info(f'DATASET={DATASET} with NUM_CLASSES={cfg.det_classe}')

    def render_frame(idx, frame, dets, tracks=False):
        from ml.vision.utils import rgb
        from torchvision.utils import draw_bounding_boxes

        if tracks:
            tids, dets = list(zip(*dets))
            dets = th.stack(dets)
            labels = [f"[{int(c)}][{tid}]" for tid, c in zip(tids, dets[:, 5])]
            colors = [rgb(tid, integral=True) for tid in tids]
            annot_dict[idx]['tracks'] = {tid: det for tid, det in zip(tids, dets.tolist())}
        else:
            labels = [DATASET[i] for i in dets[:, -1].int()]
            colors = [COLOR[i] for i in dets[:, 5].int()]
            annot_dict[idx]['detections'] = dets.tolist()
        # get boxes: x1, y1, x2, y2
        boxes = dets[:, :4]
        # draw bounding boxes
        frame = draw_bounding_boxes(frame, boxes=boxes, labels=labels, colors=colors, fill=True, width=3, font_size=25)
        return frame

    logging.info(f"Saving tracked video and frames to {export_path}")
    media = av.open(f"{export_path}/{path.stem}-tracking.mp4", 'w')
    stream = media.add_stream('h264', cfg.fps)
    stream.bit_rate = 2000000
    def track_frames(frames, start, step):
        frames = th.stack(frames)
        # Track person only
        with th.cuda.amp.autocast(enabled=cfg.det_amp):
            dets, features = detector.detect(frames, size=cfg.det_scales, conf_thres=cfg.det_cls_thres, iou_thres=cfg.det_nms_thres, batch_preprocess=True)
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
            # assert ppl_feats_f.shape[1] == 320 + 640 + 1280

            matches = tracker.update(ppls_f, ppl_feats_f.view(len(ppl_feats_f), np.prod(ppl_feats_f.shape[1:])))
            snapshot = tracker.snapshot()
            tracks = []
            for tid, info in snapshot:
                tracks.append([tid, info])
            logging.debug(f"matches[{start + (j - start) * step}]: {matches}")
            logging.debug(f"snapshot[{start + (j - start) * step}]: {snapshot}")

            # Render both dets and tracks side by side
            frame_det = frames[j - start]
            frame_trk = frame_det.clone()
            C, H, W = frame_det.shape
            idx = f'{start + (j - start) * step:03d}'
            if cfg.render_all:
                dets = th.cat([ppls_f, objs_f])
                frame_det = render_frame(idx, frame_det, dets, False)
            else:
                frame_det = render_frame(idx, frame_det, ppls_f, False)
            if tracks:
                frame_trk = render_frame(idx, frame_trk, tracks, True)

            frame = th.zeros((C, H, 2 * W), dtype=th.uint8)
            frame[:, :, :W] = frame_det
            frame[:, :, W:] = frame_trk
            write_jpeg(frame, str(export_frame / f"frame{idx}.jpg"))
            if media is not None:
                frame = av.VideoFrame.from_ndarray(frame.permute(1, 2, 0).numpy(), format='rgb24')
                packets = stream.encode(frame)
                media.mux(packets)
                logging.debug(f'Encoded: {len(packets)} {packets}, {frame}')

    frames = []
    BS = bs = cfg.batch_size
    step = 1 if src is None else fps // cfg.fps
    t = time.time()
    for i, frame in enumerate(v):
        if isinstance(frame, av.VideoFrame):
            frame = th.as_tensor(np.ascontiguousarray(frame.to_rgb().to_ndarray())).permute(2, 0, 1)
            if cfg.det_resize and cfg.det_backend in ['trt']:
                frame = TF.resize(frame, cfg.det_resize, antialias=True)
            assert frame.data.contiguous
        if i == 0:
            stream.height = frame.shape[1]
            stream.width = frame.shape[2] * 2
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
    
    # write annotations to file
    annotations = {
        'info': {
            'track': TRACK_INFO,
            'detection': DETECTION_INFO
        },
        'annotations': annot_dict
    }
    with open(f'{export_path}/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)
    
    elapse = time.time() - t
    logging.info(f"Done tracking {path.name} in {elapse:.3f}s at {(i + 1) / step / elapse:.2f}fps")
