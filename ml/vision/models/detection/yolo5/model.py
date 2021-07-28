import sys
import torch
from ml import nn, hub
from ml.nn import functional as F

GITHUB = dict(
    owner='ultralytics',
    project='yolov5',
    tag='v5.0',
)

TAGS = {
    'v1.0': '5e970d4',
    'v2.0': 'v2.0',
    'v3.0': 'd0f98c0',
    'v3.1': 'v3.1',
    'v4.0': 'v4.0',
    'v5.0': 'v5.0',     # required by pytorch=1.8+
}

class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

def github(tag='v5.0'):
    tag = TAGS[tag]
    return hub.github(owner=GITHUB['owner'], project=GITHUB['project'], tag=tag)

def forward_once(self, x, profile=False):
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        if profile:
            import thop
            o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
            t = torch_utils.time_synchronized()
            for _ in range(10):
                _ = m(x)
            dt.append((torch_utils.time_synchronized() - t) * 100)
            print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output

    if profile:
        print('%.1fms total' % sum(dt))
    if self.tag == 'v1.0':
        self.features = [y[i] for i in (17, 21, 25)]
    elif self.tag in ['v2.0', 'v3.0', 'v3.1', 'v4.0', 'v5.0']:
        self.features = [y[i] for i in (17, 20, 23)]
    else:
        raise ValueError(f"Unsupported version: {GITHUB}")
    # return x if self.training else x[0]
    return x

def from_pretrained(chkpt, model_dir=None, force_reload=False, **kwargs):
    """
    Kwargs:
        bucket(str): S3 bucket name
        key(str): path in an S3 bucket
    """
    stem, suffix = chkpt.split('.')
    tag = kwargs.get('tag', 'v3.1') 
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
    for m in model.modules():
        # XXX pytorch-1.6
        if not hasattr(m, "_non_persistent_buffers_set"):
            m._non_persistent_buffers_set = set()
    return model.float().state_dict()

def yolo5(chkpt, pretrained=False, channels=3, classes=80, fuse=True, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    """
    Args:
    Kwargs:
        tag(str): git repo tag to explicitly specify a particular commit
        url(str): direct url to download checkpoint
        s3(dict): S3 source containing bucket and key to download a checkpoint from
    """
    import types
    tag = kwargs.get('tag', 'v5.0')
    modules = sys.modules.copy()
    try:
        m = hub.load(github(tag=tag), chkpt[:len('yolov5x')], False, channels, classes, force_reload=force_reload)
        m.tag = tag
        if pretrained:
            state_dict = from_pretrained(f'{chkpt}.pt', model_dir=model_dir, force_reload=force_reload, **kwargs)
            state_dict = { k: v for k, v in state_dict.items() if m.state_dict()[k].shape == v.shape }
            m.load_state_dict(state_dict, strict=not False)
        for module in m.modules():
            # XXX export friendly for ONNX/TRT
            if isinstance(getattr(module, 'act', None), nn.Hardswish):
                module.act = Hardswish()
        m.forward_once = types.MethodType(forward_once, m)
        if tag in ['v2.0', 'v3.0', 'v3.1', 'v4.0', 'v5.0']:
            [m.save.append(layer) for layer in (17, 20, 23) if layer not in m.save]
        elif tag == 'v1.0':
            [m.save.append(layer) for layer in (17, 21, 25) if layer not in m.save]
        fuse and m.fuse()
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