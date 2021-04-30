import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import ml.models as models
import ml.nn.functional as MF
from ml.nn import SSIMLoss

from ml.utils import *
from . import *

__all__ = ['AE', 'Encoder', 'Decoder', 'VAE', 'GMM']

#EPS=1e-04 # TF: 1e-03 => safe but higher loss, PTH: 1e-05 => fail at the end

class VAE(Model):
    r"""
        To be attached to an existing encoder.
    """

    def __init__(self, cfg, nf, nz=None):
        super(VAE, self).__init__(cfg)
        nz = nz if nz is not None else nf
        self.linears = nn.ModuleList([nn.Linear(nf, nz), nn.Linear(nf, nz)])
        self.nf = nf
        self.nz = nz

    def forward(self, x):
        x = x.view(x.size(0), -1)   # Nxnf
        mu, logvar = self.linears   # Nxnz
        mu = mu(x)
        logvar = logvar(x)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            output = eps.mul(std).add_(mu)
        else:
            output = mu
        output = output.view(output.size(0), -1, 1, 1)
        return output, mu, logvar

    def loss(self, mu, logvar, sizeAverage=True, reduce=True):
        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        """

        L = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # Nxnz => N
        if reduce:
            # Paper default: reconstruction + KL divergence losses summed over all elements and batch
            return L.mean() if sizeAverage else L.sum()
        else:
            # Pointwise
            return L

class GMM(Model):
    def __init__(self, cfg, nf, D, K, beta=0.1, lrelu=0.01, dp=0.5):
        super(GMM, self).__init__(cfg)
        self.register_buffer('mu', torch.zeros(K, 1, nf))
        self.register_buffer('cov', torch.zeros(K, 1, nf)) # diagonal variances only
        self.mlp = nn.Sequential(
            nn.Linear(nf, D),
            nn.LeakyReLU(lrelu, True) if lrelu > 0 else nn.ReLU(True),
            nn.Dropout(0.5, False),
            nn.Linear(D, K),
            nn.Softmax(dim=1)
        )
        
        self.nf = nf
        self.D = D
        self.K = K
        self.beta = beta

    def forward(self, x):
        # GMM posterior gamma 
        #print('GMM.forward() x.size():', x.size(), self.nf)
        return self.mlp(x.view(x.size(0), -1))

    def loss(self, gamma, x, sizeAverage=True, reduce=True):
        N = gamma.size(0)                           # NxK
        X = x.view(-1, self.nf)                     # Nx1x1xnf => Nxnf
        X = X.repeat(self.K, 1, 1)                  # K tiles of Nxnf = KxNxnf
        gammas = gamma.sum(dim=0, keepdim=True)     # NxK => 1xK

        if self.training:
            gammaT  = gamma.t().unsqueeze(1)        # Kx1xN
            gammasT = gammas.t().unsqueeze(2)       # Kx1x1
            mu = gammaT @ X / gammasT               # Kx1xN @ KxNxnf / Kx1x1 = Kx1xnf

            # Full Covariance Matrix
            # diff = (X - mu)                       # KxNxnf - Kx1xnf = KxNxnf
            # diffT = diff.transpose(1, 2)          # KxnfxN
            # cov = gammaT * diffT @ diff / gammasT # Kx1xN * KxnfxN @ KxNxnf / Kx1x1 = Kxnfxnf

            # Variance Matrix
            cov = gammaT @ (X - mu).pow_(2) / gammasT    # Kx1xN @ KxNxnf / Kx1x1 = Kx1xnf

            # Running average
            with torch.no_grad():
                self.mu.copy_(self.beta * mu.data + (1 - self.beta) * self.mu)
                self.cov.copy_(self.beta * cov.data + (1 - self.beta) * self.cov)
        else:
            mu, cov = self.mu, self.cov
        
        # E(z[i])   = -log sum_k( phi[k] / sqrt(|2*pi * cov[k]|)  * exp( -1/2 * ((X[i] - mu[k]).pow(2) * 1/cov[k]).sum() ))
        #           = -log sum_k( exp( log(phi[k] / sqrt(|2*pi * cov[k]|)) )  * exp(    -1/2 * ((X[i] - mu[k]).pow(2) @ (1/cov[k]).transpose(0,1)) ))
        #           = -log sum_k( exp( log(phi[k] / sqrt(|2*pi * cov[k]|))              -1/2 * ((X[i] - mu[k]).pow(2) @ (1/cov[k]).transpose(0,1)) ))
        #           = -log sum_k( exp( log(phi[k]) - 1/2 * log(|2*pi * cov[k]|)         -1/2 * ((X[i] - mu[k]).pow(2) @ (1/cov[k]).transpose(0,1)) ))
        #           = -log sum_k( exp( log(phi[k]) - 1/2 * (2*pi*cov[k]).log().sum()    -1/2 * ((X[i] - mu[k]).pow(2) @ (1/cov[k]).transpose(0,1)) ))
        #           = -logsumexp(    phi.log().t() - 1/2 * (2*pi*cov).log().sum(dim=2)) -1/2 * ((X[i] - mu).pow(2) @ (1/cov).transpose(1,2))  )
        # E(z)      = -logsumexp(   (phi.log().t() - 1/2 * (2*pi*cov).log().sum(dim=2)).repeat(N, 1, 1) -1/2 * ((X - mu).pow(2) @ (1/cov).transpose(1,2)).transpose(1,0), dim=1)
        phis        = (gammas / N).log_().t()                                           # 1xK => Kx1
        dets        = -1/2 * (self.nf * math.log(2*math.pi) + cov.log().sum(dim=2))     # Kx1
        priors      = phis if (phis.data + dets.data > 0).sum() else phis + dets        # Kx1
        normals     = -1/2 * ((X - mu).pow_(2) @ (1/cov).transpose(1,2)).transpose(1,0) # KxNxnf @ Kxnfx1 = KxNx1 => NxKx1
        terms       = priors.repeat(N, 1, 1) + normals                                  # NxKx1
        E           = -MF.logsumexp(terms, dim=1)                                       # NxKx1 => Nx1
        P           = (1 / cov).sum()                                                   # 1x1
        
        """
        print(f'X[0][0]={X[0][0]}')
        print(f'mu[0]={mu[0]}')
        print(f'X[0][0] - mu[0]={X[0][0] - mu[0]}')
        print(f'phi={phi}')
        print(f'cov[0]={cov[0]}')
        print(f'1/cov[0]={1 / cov[0]}')

        print(f'phis={phis.data}')
        print(f'dets={dets.data}')
        print(f'priors={priors.data}')
        print(f'normals[0]={normals[0].data}')
        print(f'terms[0]={terms[0].data}')
        print(f'P={P}')
        #assert( priors[0].data[0] + normals[0][0].data[0] <= 0 )
        """

        if reduce:
            E   = E.mean() if sizeAverage else E.sum()   # 1x1
        else:
            P   = P.expand_as(E)                                    # 1x1 => Nx1
        return E, P

class Encoder(nn.Sequential):
    r"""
        A plain convolutional encoder.
        patch_sz=None, center_sz=None, rec_sz=None, nc=None, nf=None, nz=None, bn_eps=None, lrelu=None, dp=None,

    """

    def __init__(self, cfg, **kwargs):
        super(Encoder, self).__init__()

        # defaults from cfg
        self.cfg        = cfg
        self.ctx_sz     = cfg.ctx_sz
        self.ctx_nc     = cfg.ctx_nc 
        self.patch_sz   = cfg.patch_sz
        self.center_sz  = cfg.center_sz
        self.rec_sz     = cfg.rec_sz
        self.nc         = cfg.nc
        self.nf         = cfg.nf
        self.nz         = cfg.nz
        self.bn_eps     = cfg.bn_eps
        self.lrelu      = cfg.lrelu
        self.dp         = cfg.dp
        self.vae        = cfg.vae
        
        # overwrite if provided
        self.__dict__.update(kwargs)

        # number of encoder convolutions dividing by log2(pactch_sz)
        nc = self.ctx_nc * self.nc
        nf = self.ctx_nc * self.nf
        sz = self.ctx_sz if cfg.mask == 'stack' else self.patch_sz
        nb_dividing_filters = int(math.log2(sz // 8))  # 1st(1/2) and last(1/4)
     
        # encoder: output planes nc)->nf->2nf->4nf->8nf->100(nz)
        self.add_module('enc_conv0', nn.Conv2d(nc, nf, 4, 2, 1))  # 1/4 size
        self.add_module('enc_relu0', nn.LeakyReLU(self.lrelu, True)) if self.lrelu > 0 else self.add_module(nn.ReLU(True))
        if self.dp > 0:
            self.add_module('enc_dropout0', nn.Dropout2d(self.dp, True))
            
        filters_in  = nf
        filters_out = 2 * nf
        for i in  range(nb_dividing_filters):
            self.add_module(f'enc_conv{i+1}', nn.Conv2d(filters_in, filters_out, 4, 2, 1))  # 1/4 size
            self.add_module(f'enc_batchnorm{i+1}', nn.BatchNorm2d(filters_out, eps=self.bn_eps))
            self.add_module(f'enc_relu{i+1}', nn.LeakyReLU(self.lrelu, True)) if self.lrelu > 0 else self.add_module(nn.ReLU(True))
            if self.dp > 0:
                self.add_module(f'dropout{i+1}', nn.Dropout2d(self.dp, True))
            filters_in   = filters_out
            filters_out *= 2

        # bottleneck size should be 1x1xnz (1x1x100)
        self.add_module(f'enc_conv{nb_dividing_filters+1}', nn.Conv2d(filters_in, self.nz, 4, 4))
        if self.vae:
            self.add_module(f'enc_vae', VAE(self.cfg, self.nz, self.nz))

class Decoder(nn.Sequential):
    r"""
        A plain deconvolutional decoder.
        patch_sz=None, center_sz=None, rec_sz=None, nc=None, nf=None, nz=None, bn_eps=None, lrelu=None, dp=None,
    """

    def __init__(self, cfg, **kwargs):
        super(Decoder, self).__init__()

        # defaults from cfg
        self.cfg        = cfg
        self.ctx_sz     = cfg.ctx_sz
        self.ctx_nc     = cfg.ctx_nc 
        self.patch_sz   = cfg.patch_sz
        self.center_sz  = cfg.center_sz
        self.rec_sz     = cfg.rec_sz
        self.nc         = cfg.nc
        self.nf         = cfg.nf
        self.nz         = cfg.nz
        self.bn_eps     = cfg.bn_eps
        self.lrelu      = cfg.lrelu
        self.dp         = cfg.dp
        
        # overwrite if provided
        self.__dict__.update(kwargs)

        # number of deconvolutions in decoder (except the first and last ones), that double the size of the image
        nc = self.ctx_nc * self.nc
        nf = self.ctx_nc * self.nf
        nb_doubling_filters = int(math.log2(self.rec_sz // 2**3))   # 1st(4=2^2) and last(2)
        filters_out = nf * (2 ** nb_doubling_filters) # 8 * 2^2 = 32

        # decoder: 100(nz)->32->16->8(nf)->3
        self.add_module(f'dec_conv{0}', nn.ConvTranspose2d(self.nz, filters_out, 4, 4))
        self.add_module(f'dec_batchnorm{0}', nn.BatchNorm2d(filters_out, eps=self.bn_eps))
        self.add_module(f'dec_lrelu{0}', nn.LeakyReLU(self.lrelu, True)) if self.lrelu > 0 else self.add_module(nn.ReLU(True))
        for i in range(nb_doubling_filters):
            filters_in   = filters_out
            filters_out //= 2
            self.add_module(f'dec_conv{i+1}', nn.ConvTranspose2d(filters_in, filters_out, 4, 2, 1))
            self.add_module(f'dec_batchnorm{i+1}', nn.BatchNorm2d(filters_out, eps=self.bn_eps))
            self.add_module(f'dec_lrelu{i+1}', nn.LeakyReLU(self.lrelu, True)) if self.lrelu > 0 else self.add_module(nn.ReLU(True))
        self.add_module(f'dec_conv{nb_doubling_filters+1}', nn.ConvTranspose2d(nf, self.nc, 4, 2, 1))
        #self.cfg.loss == 'BCE' and self.add_module('sigmoid', nn.Sigmoid())
        self.add_module('sigmoid', nn.Sigmoid()) # 0 <= pixel <= 1

class AE(Model):
    r"""
        kwargs::
        patch_sz=None, center_sz=None, rec_sz=None, nc=None, nf=None, nz=None, bn_eps=None, lrelu=None, dp=None,
        vae=None, gmm=None,
    """
    def __init__(self, cfg, **kwargs):
        super(AE, self).__init__(cfg)

        # defaults from cfg
        self.c          = cfg.c
        self.ctx        = cfg.ctx
        self.ctx_sz     = cfg.ctx_sz
        self.ctx_nc     = cfg.ctx_nc 
        self.patch_sz   = cfg.patch_sz
        self.center_sz  = cfg.center_sz
        self.rec_sz     = cfg.rec_sz
        self.nc         = cfg.nc
        self.nf         = cfg.nf
        self.nz         = cfg.nz
        self.bn_eps     = cfg.bn_eps
        self.lrelu      = cfg.lrelu
        self.dp         = cfg.dp
        self.vae        = cfg.vae
        self.gmm_beta   = cfg.gmm_beta
        self.gmm_K      = cfg.gmm_K
        self.gmm_D      = cfg.gmm_D
        
        # overwrite if provided
        self.__dict__.update(kwargs)

        # constructed by subclass
        self.enc = None
        self.dec = None
        self.gmm = None

        # reconstruction loss
        self.criterion = None
        if cfg.loss == 'SSIM':
            self.criterion = SSIMLoss(cfg.ssim_kw)
        elif cfg.loss == 'MSE':
            self.criterion = F.mse_loss
        elif cfg.loss == 'BCE':
            self.criterion = F.binary_cross_entropy

    def parallelize(self, cuda=None, distributed=None):
        cuda = cuda if cuda is not None else self.cfg.nGPU > 0
        distributed = distributed if distributed is not None else self.cfg.distributed
        self.enc = nn.ModuleList([models.parallelize(enc, cuda, distributed) for enc in self.enc])
        if self.dec:
            self.dec = nn.ModuleList([models.parallelize(dec, cuda, distributed) for dec in self.dec])
        if self.gmm:
            self.gmm = nn.ModuleList([models.parallelize(gmm, cuda, distributed) for gmm in self.gmm])
        return self
    
    def forward(self, inputs, targets=None):
        r"""
            Inputs/targets by context.
        """
        inputs = isinstance(inputs, list) and inputs or [inputs]
        outputs = []
        for c, x in enumerate(inputs):
            encoding = self.enc[c](x)
            bn, mu, logvar  = encoding if self.vae else encoding, None, None
            rec, gamma, x   = bn, None, None
            if self.dec:
                rec = self.dec[c](bn)
                if self.gmm:
                    # GMM input 
                    N = bn.size(0)
                    bnV = bn.view(N, -1)
                    targetsV = targets.view(N, -1)
                    recV = rec.view(N, -1)
                    #assert((recV >= 0).data.sum() == recV.data.numel())
                    #assert((recV <= 1).data.sum() == recV.data.numel())

                    #print(f'bn.shape={bn.shape}, bnV.shape={bnV.shape}')
                    #print(f'targets.shape={targets.shape}, targetsV.shape={targetsV.shape}')

                    #ssim = MF.ssim_loss(rec, targets, window_size=self.cfg.ssim_kw, reduce=False).view(N, -1)
                    loss = self.criterion(rec, targets, size_average=True, reduce=False).view(N, -1)
                    #assert((ssim >= 0).data.sum() == ssim.data.numel())
                    #assert((ssim <= 1).data.sum() == ssim.data.numel())
                    #assert((loss == ssim).data.sum() == ssim.data.numel())
                    #print(f'ssim={ssim[-10:]}')
                    #print(f'loss={loss[-10:]}')

                    #print(f'ssim loss: {ssim[-10:].data}')
                    cos = F.cosine_similarity(recV, targetsV).view(N, -1)
                    #mse = (targetsV - recV).norm(2, 1, keepdim=True) # / targetsV.norm(2, 1, keepdim=True)
                    #print(f'mse[0]={mse[0]} of {mse.shape}, cos[0]={cos[0]} of {cos.shape}, ssim[0]={ssim[0]} of {ssim.shape}')
                    #print(f'bnV: {bnV.shape}, mse: {mse.shape}, cos: {cos.shape}, ssim: {ssim.shape}')
                    #x = torch.cat([bnV, ssim, cos, mse], dim=1)
                    x = torch.cat([bnV, loss, cos], dim=1)
                    gamma = self.gmm[c](x)
                    #print('x.size():', x.size())
                    #print('gamma.size():', gamma.size())
            outputs.append(((rec, mu, logvar), (gamma, x)))
        return outputs

    def loss(self, outputs, targets, sizeAverage=True, reduce=True):
        """Compute reconstruction losses per ctx and pairwise losses
        """
        losses  = []
        outputs = isinstance(outputs, list) and outputs or [outputs]
        zero    = torch.tensor(0.0, device=targets.device)
        for c, output in enumerate(outputs):
            (rec, mu, logvar), (gamma, x) = output
            loss = self.criterion(rec, targets, size_average=sizeAverage, reduce=reduce)
            vae  = self.cfg.vae_lambda * self.vae[c].loss(mu, logvar, sizeAverage, reduce) if self.vae else zero.expand_as(loss)
            E, P = self.gmm[c].loss(gamma, x, sizeAverage, reduce) if self.gmm else (zero.expand_as(loss), zero.expand_as(loss))
            gmm  = (self.cfg.gmm_lambda1 * E, self.cfg.gmm_lambda2 * P)
            losses.append((loss, vae, gmm))

        if len(outputs) > 1:
            # pairwise rec loss
            loss = zero
            for recs in pairwise(outputs):
                loss = loss + self.criterion(recs[0][0][0], recs[1][0][0], size_average=sizeAverage, reduce=reduce)
            losses.append(loss)
        return losses
