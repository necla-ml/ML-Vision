import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.utils.model_zoo as model_zoo

from ml.utils import Config

from .ae import *

__all__ = ['CAE',]

#EPS=1e-04 # TF: 1e-03 => safe but higher loss, PTH: 1e-05 => fail at the end

class CAE(AE):
    r"""General muilt-context autoencder.
       
    Several variants can be sepecified in the configurations:
    
    - surrounding context mask to specify a subset of context patches

      The mask is a binary string selecting context tiles as input.

    - variational bottleneck as mu and log variance

      Sampling is applied if variantional encoding is enabled during training. 

    Args:
        cfg: application specific model defaults
        bn_eps: 8e-4 rather than 1e-5 for stability
    """

    def __init__(self, cfg, **kwargs):
        super(CAE, self).__init__(cfg, **kwargs)
        if self.nz > 0:
            self.enc    = nn.ModuleList([Encoder(cfg, **kwargs) for _ in self.ctx])
            self.dec    = nn.ModuleList([Decoder(cfg, **kwargs) for _ in self.ctx])
            self.gmm    = self.cfg.gmm_K > 0 and nn.ModuleList([GMM(self.cfg, self.nz+2, self.gmm_D, self.gmm_K, self.gmm_beta, self.lrelu, self.dp) for _ in self.ctx]) or None
        else:
            # Direct transformation without decoding
            error('Direct transformation not supported by CAE')

            if self.ctx_sz % self.rec_sz == 0:
                times = math.log2(self.ctx_sz / self.rec_sz)  # 1st(1/2) and last(1/4)
                transform = nn.Conv2d
            elif self.rec_sz % self.ctx_sz == 0:
                times = math.log2(self.rec_sz / self.ctx_sz)
                transform = nn.ConvTranspose2d
            else:
                error(f"self.ctx_sz={self.ctx_sz} and rec_sz={self.rec_sz} are not mutually divisable")
            assert(times >= 2)
