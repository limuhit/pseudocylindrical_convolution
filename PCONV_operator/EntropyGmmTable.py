import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule

class EntropyGmmTable_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, delta, mean, ntop, op):
        gid = weight.device.index
        if not weight.is_contiguous(): weight = weight.contiguous()
        if not delta.is_contiguous(): delta = delta.contiguous()
        if not mean.is_contiguous(): mean = mean.contiguous()
        outputs = op[gid].forward(weight, delta, mean, ntop)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None
    

class EntropyGmmTable(BaseOpModule):
    
    def __init__(self, nstep, bias, num_gaussian, total_region, beta=1e-6, device = 0, time_it = False):
        super(EntropyGmmTable, self).__init__(device)
        self.op = { gid : PCONV.EntropyGmmTableOp(nstep,bias,num_gaussian,total_region,beta, gid, time_it) for gid in self.device_list}
        

    def forward(self, weight, delta, mean, ntop):
        res = EntropyGmmTable_AF.apply(weight, delta, mean, ntop, self.op)
        return res

class EntropyBatchGmmTable_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, ntop, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous()
        outputs = op[gid].forward_batch(x,ntop)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None
    

class EntropyBatchGmmTable(BaseOpModule):
    
    def __init__(self, nstep, bias, num_gaussian, total_region, beta=1e-6, device = 0, time_it = False):
        super(EntropyBatchGmmTable, self).__init__(device)
        self.op = { gid : PCONV.EntropyGmmTableOp(nstep,bias,num_gaussian,total_region,beta, gid, time_it) for gid in self.device_list}

    def forward(self, x,ntop):
        res = EntropyBatchGmmTable_AF.apply(x, ntop, self.op)
        return res


if __name__ == '__main__':
    from PCONV_operator import EntropyGmm
    l,n = 8,3
    dv = 'cuda:0'
    ntop = torch.tensor([l],dtype=torch.int)
    weight_raw = torch.rand((l,n),dtype=torch.float32,device=dv)
    delta = torch.rand((l,n),dtype=torch.float32,device=dv)*3
    mean = torch.rand((l,n),dtype=torch.float32,device=dv)*8-3.5
    label = torch.randint(1,7,(l,1),dtype=torch.float32,device=dv)
    gmm = EntropyGmm(3,device=0).to(dv)
    gmm2 = EntropyBatchGmmTable(8,3.5,3,65536,device=0).to(dv)
    sf = nn.Softmax(1)
    weight = sf(weight_raw)
    p1 = gmm(weight,delta,mean,label-3.5)
    p1 = torch.exp(-p1)*65536
    x = torch.cat([weight_raw.view(2,3,2,2),delta.view(2,3,2,2),mean.view(2,3,2,2)])
    p2 = gmm2(x,ntop)
    #print(torch.sum(torch.abs(p2[1].view(-1)-weight.view(-1))))
    #print(torch.sum(torch.abs(p2[2].view(-1)-delta.view(-1))))
    r = p2.view(-1,9)
    for i in range(l):
        lb = int(label[i,0])
        print(lb,p1[i],r[i,lb+1]-r[i,lb])
    pass
    idx = label.view(-1).type(torch.long)
    print(r[torch.arange(0,l),idx+1]-r[torch.arange(0,l),idx])