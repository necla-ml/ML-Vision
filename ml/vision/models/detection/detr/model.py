import sys
from pathlib import Path

import torch
from ml import io, nn, hub, logging
from ml.nn import functional as F

GITHUB_DETR = dict(
    owner='facebookresearch',
    project='detr',
    tag='main',
)

TAGS_DETR = {
    'main': 'a54b778',    # 11/15/2020
    'v0.2': '14602a7',      # 06/29/2020
}

GITHUB_DEFORMABLE_DETR = dict(
    owner='necla-ml',
    project='Deformable-DETR',
    tag='main',
)

TAGS_DEFORMABLE_DETR = {
    'main': '61854d0'
}

def github(tag='main', deformable=False):
    if deformable:
        tag = TAGS_DEFORMABLE_DETR[tag]
        return hub.github(owner=GITHUB_DEFORMABLE_DETR['owner'], project=GITHUB_DEFORMABLE_DETR['project'], tag=tag)
    else:
        tag = TAGS_DETR[tag]
        return hub.github(owner=GITHUB_DETR['owner'], project=GITHUB_DETR['project'], tag=tag)

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
    return chkpt['model']

def detr(pretrained=False, deformable=False, backbone='resnet50', num_classes=91, model_dir=None, force_reload=False, unload_after=False, **kwargs):
    """
    Kwargs:
        pretrained(bool, str): True for official checkpoint or path(str) to load a custom checkpoint
        deformable(bool): deformable successor or original
        backbone(str): feature extraction backbone architecture
        num_classes(int): number of classes to classify

        tag(str): git repo tag to explicitly specify a particular commit
        url(str): direct url to download checkpoint
        s3(dict): S3 source containing bucket and key to download a checkpoint from
        variant(str): deformable DETR variant
        panoptic(bool):
        threshold(float):
    """
    if deformable:
        '''
        Deformable DETR from GDrive
        - [x] standard: https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view?usp=sharing
        - [ ] singel scale: https://drive.google.com/file/d/1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU/view?usp=sharing
        - [ ] single scale + DC5: https://drive.google.com/file/d/1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk/view?usp=sharing
        - [ ] refinement: https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view?usp=sharing
        - [ ] two-stage: https://drive.google.com/file/d/15I03A7hNTpwuLNdfuEmW9_taZMNVssEp/view?usp=sharing
        '''
        VARIANTS = dict(
            standard='1nDWZWHuRwtwGden77NLM9JoWe-YisJnA',
            single='1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU',
            single_dc5='1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk',
            refinement='1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI',
            two_stage='15I03A7hNTpwuLNdfuEmW9_taZMNVssEp',
        )
        variant = kwargs.get('variant', 'standard')
        tag = kwargs.get('tag', GITHUB_DEFORMABLE_DETR['tag'])
        modules = sys.modules.copy()
        panoptic = kwargs.get('panoptic', False)
        backbone = backbone == 'resnet50' and 'r50' or backbone
        entry = f"deformable_detr_{backbone.replace('resnet', 'r')}{panoptic and '_panoptic' or ''}"
        m = None
        try:
            logging.info(f"Creating Deformable DETR '{entry}'")
            if panoptic:
                return_postprocessors = kwargs.get('return_postprocessors', False)
                m = hub.load(github(tag=tag, deformable=deformable), 
                             entry, 
                             num_classes=num_classes, 
                             return_postprocessor=return_postprocessors, 
                             force_reload=force_reload)
            else:
                m = hub.load(github(tag=tag, deformable=deformable), 
                             entry, 
                             num_classes=num_classes, 
                             force_reload=force_reload)
            m.tag = tag
            if pretrained:
                if isinstance(pretrained, bool):
                    # official pretrained
                    state_dict = from_pretrained(f"{entry}.pt", force_reload=force_reload, gdrive=dict(id=VARIANTS[variant]))
                    m.load_state_dict(state_dict, strict=not False)
                else:
                    # custom checkpoint
                    path = Path(pretrained)
                    if not path.exists():
                        path = f"{hub.get_dir()}/{pretrained}"
                    state_dict = io.load(path)
                    state_dict = {k: v for k, v in state_dict.items() if m.state_dict()[k].shape == v.shape}
                    m.load_state_dict(state_dict, strict=not False)
        except Exception as e:
            logging.info(f"Failed to load '{entry}': {e}")
        finally:
            # XXX Remove newly imported modules in case of conflict with next load
            if unload_after:
                for module in sys.modules.keys() - modules.keys():
                    del sys.modules[module]
        return m
    else:
        '''
        DETR entry points from hubconf
        - detr_resnet50
        - detr_resnet50_dc5
        - detr_resnet101
        - detr_resnet50_panoptic
        - detr_resnet50_dc5_panoptic
        - def detr_resnet101_panoptic
        '''
        tag = kwargs.get('tag', GITHUB_DETR['tag'])
        modules = sys.modules.copy()
        panoptic = kwargs.get('panoptic', False)
        entry = f"detr_{backbone}{panoptic and '_panoptic' or ''}"
        m = None
        try:
            logging.info(f"Create DETR arch={entry}")
            if panoptic:
                threshold = kwargs.get('threshold', 0.85)
                return_postprocessor = kwargs.get('return_postprocessor', False)
                m = hub.load(github(tag=tag), 
                             entry, 
                             pretrained=pretrained and isinstance(pretrained, bool), 
                             num_classes=num_classes, 
                             threshold=threshold, 
                             return_postprocessor=return_postprocessor, 
                             force_reload=force_reload)
            else:
                m = hub.load(github(tag=tag), 
                             entry, 
                             pretrained=pretrained and isinstance(pretrained, bool), 
                             num_classes=num_classes, 
                             force_reload=force_reload)
                logging.info(f"Loaded {'pretrained' if pretrained and isinstance(pretrained, bool) else ''} '{entry}'")
            m.tag = tag
            if isinstance(pretrained, str):
                # custom checkpoint
                path = Path(pretrained)
                if not path.exists():
                    path = f"{hub.get_dir()}/{pretrained}"
                state_dict = io.load(path)
                state_dict = {k: v for k, v in state_dict.items() if m.state_dict()[k].shape == v.shape}
                m.load_state_dict(state_dict, strict=not False)
                logging.info(f"Loaded custom pretrained '{path}'")
        except Exception as e:
            logging.info(f"Failed to load '{entry}': {e}")
        finally:
            # XXX Remove newly imported modules in case of conflict with next load
            if unload_after:
                for module in sys.modules.keys() - modules.keys():
                    del sys.modules[module]
        return m
