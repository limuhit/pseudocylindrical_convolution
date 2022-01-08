import torch
from torch import nn

class Extract_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, dims):
        y = x[:,:dims].contiguous()
        ctx.save_for_backward(x)
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        y = torch.zeros_like(x)
        dims = grad_output.shape[1]
        y[:,:dims] = grad_output
        return y, None

class StubMask(nn.Module):

    def __init__(self, dims=192):
        super(StubMask,self).__init__()
        self.dims = dims
        self.mask = None

    def setup_mask(self,x):
        if not self.mask is None:
            n,c,h,w = x.shape
            tn,tc,th,tw = self.mask.shape
            if tn == n and tc == c and th == h and tw == w:
                if x.device == self.mask.device:
                    return 
                self.mask=self.mask.to(x.device)
                return 
        self.mask = torch.ones_like(x)
        self.mask[:,self.dims:] = 0

    def forward(self,x):
        self.setup_mask(x)
        return self.mask

class Extract(nn.Module):

    def __init__(self, dims):
        super(Extract,self).__init__()
        self.dims = dims

    def forward(self,x):
        return Extract_AF.apply(x,self.dims)

if __name__ == '__main__':
    from torch.autograd import Variable
    nd,cd,hd,wd = 2,16,4,4
    data = Variable(torch.rand((nd,cd,hd,wd),dtype=torch.float32),requires_grad=True).to("cuda:0")
    data.retain_grad()
    #mk = StubMask(12)
    mk = Extract(12)
    y = mk(data)
    loss = torch.sum(y**2/2)
    loss.backward()
    #print(torch.sum(torch.abs(data[:,:12]-data.grad[:,:12])))
    #print(torch.sum(torch.abs(data[:,:12]-y)))
    pass
