import sys
import types
from pathlib import Path

import torch
from torch import hub

from ml import logging
import ml.hub as ml_hub


GITHUB_YOLOX = dict(
    owner='Megvii-BaseDetection',
    project='YOLOX',
    tag='main',
)

TAGS_YOLOX = {
    'main': 'ac58e0a5e68e57454b7b9ac822aced493b553c53'
}

def github(tag='main', deformable=False):
    tag = TAGS_YOLOX[tag]
    return ml_hub.github(owner=GITHUB_YOLOX['owner'], project=GITHUB_YOLOX['project'], tag=tag)

def custom_forward(self, x, targets=None):
    # fpn output content features of [dark3, dark4, dark5]
    fpn_outs = self.backbone(x)
    outputs = self.head(fpn_outs)

    return outputs, fpn_outs

def yolox(arch='yolox_x', pretrained=True, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    '''
    Args:
        arch (str): name of model. for example, "yolox-s", "yolox-tiny" or "yolox_custom"
        if you want to load your own model.
        pretrained (bool): load pretrained weights into the model. Default to True.
        device (str): default device to for model. Default to None.
        num_classes (int): number of model classes. Default to 80.
        exp_path (str): path to your own experiment file. Required if name="yolox_custom"
        ckpt_path (str): path to your own ckpt. Required if name="yolox_custom" and you want to
            load a pretrained model
    '''
    tag = kwargs.get('tag', GITHUB_YOLOX['tag'])
    modules = sys.modules.copy()
    m = None
    try:
        logging.info(f"Create YOLOX arch={arch}")
        m = hub.load(github(tag=tag), 
                    arch, 
                    pretrained=pretrained and isinstance(pretrained, bool), 
                    num_classes=num_classes, 
                    device=device,
                    force_reload=force_reload)
        logging.info(f"Loaded {'pretrained' if pretrained and isinstance(pretrained, bool) else ''} '{arch}'")
        m.tag = tag

        if isinstance(pretrained, str):
            # custom checkpoint
            path = Path(pretrained)
            if not path.exists():
                path = f"{hub.get_dir()}/{pretrained}"
            state_dict = torch.load(path, map_location='cpu')
            state_dict = {k: v for k, v in state_dict.items() if m.state_dict()[k].shape == v.shape}
            m.load_state_dict(state_dict, strict=True)
            logging.info(f"Loaded custom pretrained '{path}'")

        # set custom forward method to get backbone features
        forward_lst = list(filter(lambda x: x == 'forward', dir(m)))
        setattr(m, forward_lst[-1], types.MethodType(custom_forward, m))

        from yolox.exp.build import get_exp_by_name
        exp = get_exp_by_name(arch)
    
    except Exception as e:
        logging.info(f"Failed to load '{arch}': {e}")
        raise e
    
    finally:
        # XXX Remove newly imported modules in case of conflict with next load
        if unload_after:
            for module in sys.modules.keys() - modules.keys():
                del sys.modules[module]

    m.to('cpu')
    return m, exp