import sys
from pathlib import Path

import torch
from ml import io, nn, hub, logging
from ml.nn import functional as F

GITHUB = dict(
    owner='necla-ml',
    project='Lite-HRNet',
    tag='main',
)

TAGS = {
    'main': 'cdef7bc',
    'hrnet': '7b9049d',
}

def github(tag='main'):
    tag = TAGS[tag]
    return hub.github(owner=GITHUB['owner'], project=GITHUB['project'], tag=tag)

def from_pretrained(chkpt, model_dir=None, force_reload=False, **kwargs):
    # TODO naming for custom checkpoints
    """
    Kwargs:
        owner(str): github owner
        project(str): github project
        tag(str): github branch/tag
        bucket(str): S3 bucket name
        key(str): path in an S3 bucket
    Returns:
        state_dict:
    """
    stem, suffix = chkpt.split('.')
    tag = kwargs.get('tag', 'main') 
    filename = f"{stem}-{tag}.{suffix}"
    s3 = kwargs.get('s3', None)
    gdrive = kwargs.get('gdrive', None)
    if gdrive and 'id' in gdrive:
        chkpt = hub.load_state_dict_from_gdrive(gdrive['id'], filename, model_dir=None, map_location=None, force_reload=False, progress=True, check_hash=False)
    elif s3 and s3.get('bucket', None) and s3.get('key', None):
        url = f"s3://{s3['bucket']}/{s3['key']}"
        # logging.info(f"Loading chkpt from url={url} to filename={stem}-{tag}.{suffix}, s3={s3}")
        chkpt = hub.load_state_dict_from_url(url, 
                                             model_dir=model_dir, 
                                             map_location=torch.device('cpu'),
                                             file_name=filename,
                                             force_reload=force_reload)
    else:
        # GitHub Release
        owner = kwargs.get('owner', GITHUB_DETR['owner'])
        proj = kwargs.get('project', GITHUB_DETR['project'])
        url = hub.github_release_url(owner, proj, tag, chkpt)
        chkpt = hub.load_state_dict_from_url(url, 
                                             model_dir=model_dir, 
                                             map_location=torch.device('cpu'),
                                             file_name=filename,
                                             force_reload=force_reload)

    '''
    model = chkpt['model']
    for m in model.modules():
        # XXX pytorch-1.6
        if not hasattr(m, "_non_persistent_buffers_set"):
            m._non_persistent_buffers_set = set()
    return model.float().state_dict()
    '''
    # dict_keys(['meta', 'state_dict', 'optimizer'])
    return chkpt['state_dict']

def posenet(pretrained=False, arch='litehrnet_30_coco_384x288', model_dir=None, force_reload=False, unload_after=False, **kwargs):
    """
    Kwargs:
        pretrained(bool, str): True for official checkpoint or path(str) to load a custom checkpoint

        tag(str): git repo tag to explicitly specify a particular commit
        url(str): direct url to download checkpoint
        s3(dict): S3 source containing bucket and key to download a checkpoint from
        threshold(float):
    
    """
    ARCH = dict(litehrnet_18_coco_256x192='1ZewlvpncTvahbqcCFb-95C3NHet30mk5',
                litehrnet_18_coco_384x288='1E3S18YbUfBm7YtxYOV7I9FmrntnlFKCp',
                litehrnet_30_coco_256x192='1KLjNInzFfmZWSbEQwx-zbyaBiLB7SnEj',
                litehrnet_30_coco_384x288='1BcHnLka4FWiXRmPnJgJKmsSuXXqN4dgn',
                litehrnet_18_mpii_256x256='1bcnn5Ic2-FiSNqYOqLd1mOfQchAz_oCf',
                litehrnet_30_mpii_256x256='1JB9LOwkuz5OUtry0IQqXammFuCrGvlEd')

    tag = kwargs.get('tag', GITHUB['tag'])
    modules = sys.modules.copy()
    entry = 'posenet'
    m = None
    try:
        logging.info(f"Creating '{entry}(arch={arch})'")
        m = hub.load(github(tag=tag), 
                     entry,
                     arch,
                     force_reload=force_reload)
        m.tag = tag
        if pretrained:
            if isinstance(pretrained, bool):
                # official pretrained
                state_dict = from_pretrained(f"{entry}.pt", force_reload=force_reload, gdrive=dict(id=ARCH[arch]))
            else:
                # custom checkpoint
                path = Path(pretrained)
                if not path.exists():
                    path = f"{hub.get_dir()}/{pretrained}"
                state_dict = io.load(path, map_location='cpu')
                state_dict = {k: v for k, v in state_dict.items() if m.state_dict()[k].shape == v.shape}
            m.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logging.info(f"Failed to load '{entry}': {e}")
    finally:
        # XXX Remove newly imported modules in case of conflict with next load
        if unload_after:
            for module in sys.modules.keys() - modules.keys():
                del sys.modules[module]
    m.to('cpu')
    return m