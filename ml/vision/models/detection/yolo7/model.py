import os
import sys

import torch
from ml import hub

GITHUB = dict(
    owner='WongKinYiu',
    project='yolov7',
    tag='main',
)

TAGS = {
    'main': '55b90e111984dd85e7eed327e9ff271222aa8b82',
}

FEATURE_LAYERS = {
    'yolov7-tiny': (57, 65, 73),
    'yolov7': (75, 88, 101),
    'yolov7x': (87, 102, 117),
    'yolov7-w6': (93, 103, 113),
    'yolov7-e6': (111, 123, 135),
    'yolov7-d6': (129, 143, 157),
    'yolov7-e6e': (210, 233, 256),
}

def github(tag='main'):
    tag = TAGS[tag]
    return hub.github(owner=GITHUB['owner'], project=GITHUB['project'], tag=tag)

def forward_once(self, x, profile=False, visualize=False):
    """ Collect features for tracking """
    y = [] # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output

    self.features = [y[i] for i in FEATURE_LAYERS[self.chkpt]]

    return x

def from_pretrained(chkpt, model_dir=None, force_reload=False, **kwargs):
    """
    Kwargs:
        bucket(str): S3 bucket name
        key(str): path in an S3 bucket
    """
    stem, suffix = chkpt.split('.')
    tag = kwargs.get('tag', 'main') 
    s3 = kwargs.get('s3', None)
    if s3 and s3.get('bucket', None) and s3.get('key', None):
        url = f"s3://{s3['bucket']}/{s3['key']}"
    else:
        # GitHub Release
        # NOTE: checkpoints assets only available in released version
        # but v0.1 release is not TRT compatible so using main: 55b90e111984dd85e7eed327e9ff271222aa8b82 
        # as the default 
        url = hub.github_release_url(GITHUB['owner'], GITHUB['project'], 'v0.1', chkpt)

    print(f"Loading chkpt from url={url} to filename={stem}.{suffix}, s3={s3}")
    chkpt = hub.load_state_dict_from_url(url, 
                                         model_dir=model_dir, 
                                         map_location=torch.device('cpu'),
                                         file_name=f'{stem}.{suffix}',
                                         force_reload=force_reload)

    model = chkpt['model']
    return model.float()

def yolo7(chkpt, pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, autoshape=False, **kwargs):
    """
    Args:
    Kwargs:
        tag(str): git repo tag to explicitly specify a particular commit
        url(str): direct url to download checkpoint
        s3(dict): S3 source containing bucket and key to download a checkpoint from
    """
    import types
    tag = kwargs.get('tag', 'main')
    modules = sys.modules.copy()
    try:
        # XXX: makes the `model` module available from og repo for torch.load
        # only yolov7 is available from hubconf so download source and then use from_pretrained to load custom model (yolov7-tiny, yolov7x, etc)
        m = hub.load(github(tag=tag), 'yolov7', pretrained=False, channels=channels, classes=classes, force_reload=force_reload, autoshape=autoshape)
        if pretrained:
            m = from_pretrained(f'{chkpt}.pt', model_dir=model_dir, force_reload=force_reload, **kwargs)
            setattr(m, 'chkpt', chkpt)

        # replace forward_once with our custom forward_once to collect features
        forward_m = list(filter(lambda x: x.endswith('forward_once'), dir(m)))
        assert forward_m, f'Cannot find forward_once method in the model'
        setattr(m, forward_m[0], types.MethodType(forward_once, m))

        # add feature layers to save list for `forward_once`
        for layer in FEATURE_LAYERS: 
            if layer not in m.save:
                m.save.append(layer) 
        # fuse if enabled
        fuse and m.fuse()
    except Exception as e:
        raise e
    finally:
        # XXX Remove newly imported modules in case of conflict with next load
        if unload_after:
            for module in sys.modules.keys() - modules.keys():
                del sys.modules[module]
    return m

def yolo7t(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7-tiny', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo7s(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo7x(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7x', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo7w6(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7-w6', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo7e6(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7-e6', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo7d6(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7-d6', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo7e6e(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo7('yolov7-e6e', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)