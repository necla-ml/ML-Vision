import sys
import torch
from ml import hub

GITHUB = dict(
    owner='ultralytics',
    project='yolov5',
    tag='v6.0',
)

TAGS = {
    'v6.0': 'v6.0'
}

FEATURE_LAYERS = (17, 20, 23)

def github(tag='v6.0'):
    tag = TAGS[tag]
    return hub.github(owner=GITHUB['owner'], project=GITHUB['project'], tag=tag)

def forward_once(self, x, profile=False, visualize=False):
    """ Collect features for tracking """
    y = []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output

    self.features = [y[i] for i in FEATURE_LAYERS]

    return x

def from_pretrained(chkpt, model_dir=None, force_reload=False, **kwargs):
    """
    Kwargs:
        bucket(str): S3 bucket name
        key(str): path in an S3 bucket
    """
    stem, suffix = chkpt.split('.')
    tag = kwargs.get('tag', 'v6.0') 
    s3 = kwargs.get('s3', None)
    if s3 and s3.get('bucket', None) and s3.get('key', None):
        url = f"s3://{s3['bucket']}/{s3['key']}"
    else:
        # GitHub Release
        url = hub.github_release_url('ultralytics', 'yolov5', tag, chkpt)

    # logging.info(f"Loading chkpt from url={url} to filename={stem}-{tag}.{suffix}, s3={s3}")
    chkpt = hub.load_state_dict_from_url(url, 
                                         model_dir=model_dir, 
                                         map_location=torch.device('cpu'),
                                         file_name=f'{stem}-{tag}.{suffix}',
                                         force_reload=force_reload)

    model = chkpt['model']
    from models.yolo import Detect
    for m in model.modules():
        # new Detect Layer compatibility
        if type(m) is Detect and not isinstance(m.anchor_grid, list):  
            delattr(m, 'anchor_grid')
            setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)

    return model.float()

def yolo5(chkpt, pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, autoshape=False, **kwargs):
    """
    Args:
    Kwargs:
        tag(str): git repo tag to explicitly specify a particular commit
        url(str): direct url to download checkpoint
        s3(dict): S3 source containing bucket and key to download a checkpoint from
    """
    import types
    tag = kwargs.get('tag', 'v6.0')
    modules = sys.modules.copy()
    try:
        # XXX: makes the `model` module available from og repo for torch.load
        m = hub.load(github(tag=tag), chkpt[:len('yolov5x')], pretrained=True, channels=channels, classes=classes, force_reload=force_reload, autoshape=autoshape)
        if pretrained:
            m = from_pretrained(f'{chkpt}.pt', model_dir=model_dir, force_reload=force_reload, **kwargs)
            setattr(m, 'tag', tag)

        # replace forward_once with our custom forward_once
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

def yolo5l(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo5('yolov5l', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)

def yolo5x(pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    return yolo5('yolov5x', pretrained, channels, classes, fuse, model_dir, force_reload, unload_after, **kwargs)