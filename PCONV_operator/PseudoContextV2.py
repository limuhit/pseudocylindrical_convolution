import torch
import PCONV
from torch._C import device, dtype
from PCONV_operator.BaseOpModule import BaseOpModule
from PCONV_operator.GDN import LowerBound
from torch import nn
import math
from PCONV_operator.base import set_weight
from PCONV_operator.EntropyContextNew import EntropyContextNew



class PseudoContextV2(BaseOpModule):

    def __init__(self, npart, opt=True, rt=20, device=0, time_it=False):
        super(PseudoContextV2,self).__init__(device)
        weight = set_weight(npart,opt)
        self.op = { gid : PCONV.PseudoContextOp(npart, rt, weight, gid, time_it) for gid in self.device_list}
  
    def setup_context(self,w):
        for gid in self.op.keys():
            self.op[gid].start_context(w)

    def get_addr(self,gid):
        return self.op[gid].addr()

    def produce_fill_param(self, gid, h, w):
        return self.op[gid].produce_fill_param(h,w)

class PseudoEntropyContext(BaseOpModule):
    
    def __init__(self, npart, context_version=1, opt=True, rt=20, device=0, time_it=False):
        super(PseudoEntropyContext,self).__init__(device)
        weight = set_weight(npart,opt)
        self.op = { gid : PCONV.PseudoEntropyContextOp(npart, rt, context_version, weight, gid, time_it) for gid in self.device_list}
  
    def setup_context(self,w):
        for gid in self.op.keys():
            self.op[gid].start_context(w)

    def get_addr(self,gid):
        return self.op[gid].addr()

class PseudoEntropyPad_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoEntropyPad(BaseOpModule):
    
    def __init__(self, pad, npart, ctx:PseudoEntropyContext, device = 0, time_it = False):
        super(PseudoEntropyPad, self).__init__(device)
        self.op = { gid : PCONV.PseudoEntropyPadOp(pad,npart,ctx.get_addr(gid), gid, time_it) for gid in self.device_list}

    def forward(self, x):
        res = PseudoEntropyPad_AF.apply(x, self.op)
        return res


class PseudoPadV2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoPadV2(BaseOpModule):
    
    def __init__(self, pad, npart, ctx:PseudoContextV2, device = 0, time_it = False):
        super(PseudoPadV2, self).__init__(device)
        self.op = { gid : PCONV.PseudoPadOp(pad,npart,ctx.get_addr(gid), gid, time_it) for gid in self.device_list}
        
    def forward(self, x):
        res = PseudoPadV2_AF.apply(x, self.op)
        return res

class PseudoFillV2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous() 
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None

class PseudoFillV2(BaseOpModule):
    #int pad, int npart, int fvalue, int trim, std::string addr, int context_version, int device = 0, bool timeit=false
    def __init__(self, pad, npart, ctx, fvalue=0, trim=0, device = 0, time_it = False):
        super(PseudoFillV2, self).__init__(device)
        if isinstance(ctx,PseudoContextV2):
            self.op = { gid : PCONV.PseudoFillOp(pad, npart, fvalue, trim, ctx.get_addr(gid), 0, gid, time_it) for gid in self.device_list}
        elif isinstance(ctx,PseudoEntropyContext):
            self.op = { gid : PCONV.PseudoFillOp(pad, npart, fvalue, trim, ctx.get_addr(gid), 1, gid, time_it) for gid in self.device_list}
        else:
            self.op = { gid : PCONV.PseudoFillOp(pad, npart, fvalue, trim, ctx.get_addr(gid), 2, gid, time_it) for gid in self.device_list}

    def forward(self, x):
        res = PseudoFillV2_AF.apply(x, self.op)
        return res



class PseudoGDNV2(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 npart,
                 ctx:PseudoContextV2,
                 device=0,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(PseudoGDNV2, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])
        self.trim = PseudoFillV2(0,npart,ctx,device=device)
        self.mask = None
        if isinstance(device, int):
            self.build(ch, torch.device('cuda:%d'%device))
        else:
            self.build(ch, torch.device('cuda:%d'%device[0]))
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset
  
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def setup_mask(self,x):
        
        if self.mask is not None:
            n,c,h,w = x.shape
            tn,tc,th,tw = self.mask.shape
            if tn==n and tc==c and th==h and tw==w: return
        self.mask = torch.ones_like(x).detach()
        self.mask = self.trim(self.mask)

    def forward(self, inputs):
        # Assert internal parameters to same device as input
        self.beta = self.beta.to(inputs.device)
        self.gamma = self.gamma.to(inputs.device)
        self.pedestal = self.pedestal.to(inputs.device)

        _, ch, _, _ = inputs.size()
        self.setup_mask(inputs)
        inputs = inputs * self.mask


        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
        norm_ = norm_ * self.mask + 1 - self.mask
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        return outputs

class Pseudo_QUANT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, count, quant_op, training):
        if not x.is_contiguous(): x = x.contiguous()
        gid = x.device.index
        outputs = quant_op[gid].forward(x, weight, count, training)
        ctx.save_for_backward(x, outputs[0])
        ctx.quant_op = quant_op
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs[0],outputs[1]
        
    @staticmethod
    def backward(ctx, *grad_output):
        grad_list = list(grad_output)
        grad_list = [pt if pt.is_contiguous() else pt.contiguous() for pt in grad_list]
        x,out = ctx.saved_tensors
        gid = x.device.index
        outputs  = ctx.quant_op[gid].backward(grad_list,x,out)
        return outputs[0], outputs[1], outputs[2].clone().detach(), None, None

class PseudoQUANTV2(BaseOpModule):
    
    def __init__(self, channel, bin_num, npart, ctx:PseudoContextV2, check_iters=100, weight_decay=0.9, ntop=1, top_alpha=0.1, device_id=0, time_flag=False):
        super(PseudoQUANTV2, self).__init__(device_id)
        ta = 1./(bin_num+1)
        tb = math.log(ta)
        self.weight = nn.Parameter(torch.zeros((channel,bin_num),dtype=torch.float32).to('cuda:%d'%self.device_list[0]))
        self.weight.data[:,0] = ta
        self.weight.data[:,1:] = tb
        self.count = nn.Parameter(torch.zeros((channel,bin_num),dtype=torch.float32).to('cuda:%d'%self.device_list[0]))
        self.op = {gid:PCONV.PseudoQuantOp(channel, bin_num, npart, weight_decay, check_iters, ntop, top_alpha,ctx.get_addr(gid), gid, time_flag) for gid in self.device_list}

    def forward(self, x):
        res = Pseudo_QUANT_AF.apply(x, self.weight, self.count, self.op, self.training)
        return res


class Pseudo_DQUANT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, quant_op):
        if not x.is_contiguous(): x = x.contiguous()
        gid = x.device.index
        outputs = quant_op[gid].forward(x, weight)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None

class PseudoDQUANT(BaseOpModule):
    
    def __init__(self, channel, bin_num, npart, ctx:PseudoContextV2, device_id=0, time_flag=False):
        super(PseudoDQUANT, self).__init__(device_id)
        self.weight = nn.Parameter(torch.zeros((channel,bin_num),dtype=torch.float32).to('cuda:%d'%self.device_list[0]))
        self.op = {gid:PCONV.PseudoDQuantOp(npart, channel, bin_num, ctx.get_addr(gid), gid, time_flag) for gid in self.device_list}

    def forward(self, x):
        res = Pseudo_DQUANT_AF.apply(x, self.weight, self.op)
        return res


    