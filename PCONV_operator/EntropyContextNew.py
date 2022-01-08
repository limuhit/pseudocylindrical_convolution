import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule
from PCONV_operator.base import set_weight


class EntropyContextNew(BaseOpModule):
    
    def __init__(self, npart, rt=18, opt=False, device = 0, time_it = False):
        super(EntropyContextNew, self).__init__(device)
        weight = set_weight(npart,opt)
        self.op = { gid : PCONV.EntropyContextOp(npart, rt, weight, gid, time_it) for gid in self.device_list}
        

    def setup_context(self,w):
        for gid in self.op.keys():
            self.op[gid].start_context(w)

    def get_addr(self,gid):
        return self.op[gid].addr()

class EntropyAdd_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, y, op):
        gid = x.device.index
        outputs = op[gid].forward(x,y)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None
    

class EntropyAdd(BaseOpModule):
    
    def __init__(self, npart,channel,ngroup,pad,ctx:EntropyContextNew, device = 0, time_it = False):
        super(EntropyAdd, self).__init__(device)
        self.op = { gid : PCONV.EntropyAddOp(npart,channel,ngroup,pad,ctx.get_addr(gid), gid, time_it) for gid in self.device_list}

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()    
    
    def forward(self, x,y):
        res = EntropyAdd_AF.apply(x, y, self.op)
        return res

class EntropyCtxPadRun2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None
    

class EntropyCtxPadRun2(BaseOpModule):
    
    def __init__(self, pad, npart, ngroup, ctx:EntropyContextNew, input=False, device = 0, time_it = False):
        super(EntropyCtxPadRun2, self).__init__(device)
        self.op = { gid : PCONV.EntropyCtxPadRun2Op(pad, npart, ngroup, input, ctx.get_addr(gid), gid, time_it) for gid in self.device_list}
    
    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        res = EntropyCtxPadRun2_AF.apply(x, self.op)
        return res

class DExtract2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x,op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous()
        outputs = op[gid].forward(x)
        return outputs[0], outputs[1]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None
    

class DExtract2(BaseOpModule):
    
    def __init__(self, npart, nchannel, label, ctx:EntropyContextNew, device = 0, time_it = False):
        super(DExtract2, self).__init__(device)
        self.op = { gid : PCONV.DExtract2Op(npart,nchannel,label, ctx.get_addr(gid), gid, time_it) for gid in self.device_list}

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        res = DExtract2_AF.apply(x, self.op)
        return res


class DExtract2Batch_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x,  op):
        gid = x.device.index
        outputs = op[gid].forward_batch(x)
        return outputs[0], outputs[1]
    
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None
    

class DExtract2Batch(BaseOpModule):
    
    def __init__(self, npart, nchannel, ctx:EntropyContextNew, device = 0, time_it = False):
        super(DExtract2Batch, self).__init__(device)
        self.op = { gid : PCONV.DExtract2Op(npart,nchannel,True, ctx.get_addr(gid), gid, time_it) for gid in self.device_list}

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        res = DExtract2Batch_AF.apply(x, self.op)
        return res

class DInput2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x,  op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous()
        outputs = op[gid].forward(x)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None
    

class DInput2(BaseOpModule):
    
    def __init__(self, nchannel, npart, ctx:EntropyContextNew, pad=0, bias=0, repeat=1,device = 0, time_it = False):
        super(DInput2, self).__init__(device)
        self.op = { gid : PCONV.DInput2Op(nchannel, npart, pad, bias, repeat, ctx.get_addr(gid),  gid, time_it) for gid in self.device_list}
    
    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        res = DInput2_AF.apply(x, self.op)
        return res

class EntropyConv2_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward(x, weight, bias)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class EntropyConv2Act_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, act, op):
        gid = x.device.index
        outputs = op[gid].forward_act(x, weight, bias, act)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None
    
class EntropyConv2BT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward_batch(x, weight, bias)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class EntropyConv2ActBT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, act, op):
        gid = x.device.index
        outputs = op[gid].forward_act_batch(x, weight, bias, act)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None




class EntropyConv2(BaseOpModule):
    
    def __init__(self, npart, ngroup, c_in, c_out, kernel_size, ctx:EntropyContextNew, pad_in=2, pad_out=2, hidden=False, act=True, device = 0, time_it = False):
        super(EntropyConv2, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = {gid : PCONV.EntropyConv2Op(npart,channel,ngroup,nout,kernel_size,constrain, pad_in, pad_out, ctx.get_addr(gid),  gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.rand((nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        #self.weight = nn.Parameter(torch.arange(1,nout*channel*kernel_size*kernel_size,dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((nout),dtype=torch.float32))
        self.act = act
        self.relu = nn.Parameter(torch.zeros((nout),dtype=torch.float32)) if act else None
    
    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()


    def forward(self, x):
        if self.act:
            return EntropyConv2Act_AF.apply(x, self.weight, self.bias, self.relu, self.op)
        else:
            return EntropyConv2_AF.apply(x, self.weight, self.bias, self.op)

class EntropyConv2Batch(BaseOpModule):
    
    def __init__(self, npart, ngroup, c_in, c_out, kernel_size, ctx:EntropyContextNew,  pad_in=2, pad_out=2, batch=3, hidden=False, act=True, device = 0, time_it = False):
        super(EntropyConv2Batch, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = {gid : PCONV.EntropyConv2Op(npart,channel,ngroup,nout,kernel_size,constrain, pad_in, pad_out, ctx.get_addr(gid), gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.rand((batch,nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        #self.weight = nn.Parameter(torch.arange(1,nout*channel*kernel_size*kernel_size,dtype=torch.float32))
        self.bias = nn.Parameter(torch.rand((batch,nout),dtype=torch.float32))
        self.act = act
        self.relu = nn.Parameter(torch.rand((batch,nout),dtype=torch.float32)) if act else None

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        if self.act:
            return EntropyConv2ActBT_AF.apply(x, self.weight, self.bias, self.relu, self.op)
        else:
            return EntropyConv2BT_AF.apply(x, self.weight, self.bias, self.op)


class EntropyConvD(nn.Module):
    
    def __init__(self, ngroups, cin, cout, hidden, npart, out_layer:bool, ctx:EntropyContextNew, device_id, act=True):
        super(EntropyConvD,self).__init__()
        pad_out = 0 if out_layer else 2
        self.pad = EntropyCtxPadRun2(2,npart,ngroups,ctx,not hidden,device=device_id)#PseudoCtxPad(2,npart,cin*ngroups,ctx,device=device_id)
        self.conv = EntropyConv2(npart,ngroups,cin,cout,5,ctx,2,pad_out,hidden=hidden,act=act,device=device_id)#MaskConv2(ngroups,cin,cout,5,hidden,device_id)

    def forward(self,x):
        tx = self.pad(x)
        tx = self.conv(tx)
        return tx

class EntropyResidualBlockD(nn.Module):
    
    def __init__(self, ngroups, cpn, npart, ctx:EntropyContextNew, device_id=0):
        super(EntropyResidualBlockD, self).__init__()
        self.conv1 = EntropyConvD(ngroups,cpn,cpn,True,npart,False,ctx,device_id,True)
        self.conv2 = EntropyConvD(ngroups,cpn,cpn,True,npart,False,ctx,device_id,True)
        self.add = EntropyAdd(npart,cpn*ngroups,ngroups,2,ctx,device=device_id)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        y = self.add(y,x)
        return y
