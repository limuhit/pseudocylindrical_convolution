import torch
from torch import nn
from PCONV_operator import  Dtow, SphereSlice, SphereUslice, EntropyGmm, ContextReshape, DropGrad, MaskConv2
from PCONV_operator import PseudoContextV2,  PseudoPadV2, PseudoFillV2, PseudoQUANTV2, PseudoGDNV2, PseudoEntropyContext, PseudoEntropyPad
from PCONV_operator import StubMask, Extract, ChannelGroupConv, ChannelGroupReshape

class ClipData_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x.clone().detach()
        f1 = x<0
        f2 = x>1
        y[f1] = x[f1]*0.01
        y[f2] = 1+(x[f2]-1)*0.01
        ctx.save_for_backward(f1,f2)
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        y = grad_output.clone().detach()
        f1,f2 = ctx.saved_tensors
        y[f1] = grad_output[f1]*0.01
        y[f2] = grad_output[f2]*0.01
        return y

class ClipData(nn.Module):

    def __init__(self):
        super(ClipData,self).__init__()

    def forward(self,x):
        return ClipData_AF.apply(x)

class ResidualBlock(nn.Module):
    
    def __init__(self, channels, npart, ctx:PseudoContextV2, device_id=0):
        super(ResidualBlock,self).__init__()
        self.pad = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv1 = nn.Conv2d(channels, channels//2, 1, 1)
        self.relu1 = nn.PReLU(channels//2)
        self.conv2 = nn.Conv2d(channels//2, channels//2, 3, 1)
        self.relu2 = nn.PReLU(channels//2)
        self.conv3 = nn.Conv2d(channels//2, channels, 1, 1)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x):
        tx = self.pad(x)
        y = self.relu1(self.conv1(tx))
        y = self.relu2(self.conv2(y))
        y = self.conv3(y)
        return self.trim(x+y)

class AttentionBlock(nn.Module):
    
    def __init__(self, channels, npart, ctx:PseudoContextV2, device_id = 0):
        super(AttentionBlock, self).__init__()
        self.trunk = nn.Sequential(
            ResidualBlock(channels,npart,ctx,device_id),
            ResidualBlock(channels,npart,ctx,device_id),
            ResidualBlock(channels,npart,ctx,device_id)
        )
        self.attention = nn.Sequential(
            ResidualBlock(channels,npart,ctx,device_id),
            ResidualBlock(channels,npart,ctx,device_id),
            ResidualBlock(channels,npart,ctx,device_id),
            nn.Conv2d(channels,channels,1,1,0),
            nn.Sigmoid()
        )
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
    
    def forward(self, x):
        t = self.trunk(x)
        a = self.attention(x)
        return self.trim(x + t*a)

class ResidualBlockV2(nn.Module):
    
    def __init__(self, channels, npart, ctx:PseudoContextV2, device_id):
        super(ResidualBlockV2,self).__init__()
        self.pad = PseudoPadV2(2,npart,ctx,device=device_id)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1)
        self.relu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1)
        self.relu2 = nn.PReLU(channels)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x):
        tx = self.pad(x)
        y = self.relu1(self.conv1(tx))
        y = self.relu2(self.conv2(y))
        return self.trim(x + y)

class ResidualBlockDown(nn.Module):
    
    def __init__(self, channels, channel_in, npart, ctx:PseudoContextV2, device_id):
        super(ResidualBlockDown,self).__init__()
        self.pad1 = PseudoPadV2(1,npart, ctx, device=device_id)
        self.conv1 = nn.Conv2d(channel_in, channels, 3, 2)
        self.relu1 = nn.PReLU(channels)
        self.pad2 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1)
        self.relu2 = PseudoGDNV2(channels, npart, ctx, device_id)
        self.short_cut = nn.Conv2d(channel_in, channels, 1, 2) 
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x):
        t = self.short_cut(x)
        y = self.pad1(x)
        y = self.relu1(self.conv1(y))
        y = self.pad2(y)
        y = self.relu2(self.conv2(y))
        return self.trim(t + y)

class SphereConv2(nn.Module):
    def __init__(self, channel_in, channel_out, npart, ctx:PseudoContextV2, device_id = 0):
        super(SphereConv2,self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, 3, 2, 0)
        self.pad = PseudoPadV2(1,npart,ctx,device=device_id)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
    def forward(self,x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.trim(x)
        return x


class EncoderV2(nn.Module):
    
    def __init__(self, channels, code_channels, npart, ctx:PseudoContextV2, device_id):
        super(EncoderV2, self).__init__()
        self.net = nn.Sequential(
            ResidualBlockDown(channels,3,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            ResidualBlockDown(channels,channels,npart,ctx,device_id),
            AttentionBlock(channels,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            ResidualBlockDown(channels,channels,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            SphereConv2(channels,channels,npart,ctx,device_id),
            AttentionBlock(channels, npart, ctx, device_id),
            nn.Conv2d(channels, code_channels, 1, 1),
        )
        self.act = nn.Sigmoid()
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x):
        x = self.net(x)
        code = self.act(x)
        return self.trim(code) 

class ResidualBlockUp(nn.Module):
    
    def __init__(self, channels, npart, ctx:PseudoContextV2, device_id):
        super(ResidualBlockUp,self).__init__()
        self.pad1 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv1 = nn.Conv2d(channels, channels*4, 3, 1)
        self.relu1 = nn.PReLU(channels*4)
        self.dtow1 = Dtow(2, True, version=0, device=device_id)
        self.pad2 = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1)
        self.relu2 = PseudoGDNV2(channels, npart, ctx, device_id, inverse = True)
        self.short_cut = nn.Conv2d(channels, channels*4, 1, 1)
        self.dtow2 = Dtow(2, True, version=0, device=device_id)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self, x):
        br1 = self.pad1(x)
        br1 = self.relu1(self.conv1(br1))
        br1 = self.dtow1(br1)
        br1 = self.pad2(br1)
        br1 = self.relu2(self.conv2(br1))
        br2 = self.dtow2(self.short_cut(x))
        return self.trim(br1 + br2)

class SphereConvOld(nn.Module):
    
    def __init__(self, npart, channel_in, channel_out, ctx:PseudoContextV2, device_id = 0):
        super(SphereConvOld,self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, 1, 1)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)

    def forward(self,x):
        x = self.conv(x)
        return self.trim(x)


class DecoderV2(nn.Module):
    
    def __init__(self, channels, code_channels, npart, ctx:PseudoContextV2, device_id):
        super(DecoderV2,self).__init__()
        self.net = nn.Sequential(
            SphereConvOld(npart,code_channels,channels,ctx,device_id),
            AttentionBlock(channels,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            ResidualBlockUp(channels,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            ResidualBlockUp(channels,npart,ctx,device_id),
            AttentionBlock(channels,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            ResidualBlockUp(channels,npart,ctx,device_id),
            ResidualBlockV2(channels,npart,ctx,device_id),
            PseudoPadV2(1,npart,ctx,device=device_id),
            nn.Conv2d(channels, 12, 3, 1),
            Dtow(2, True, version=0, device=device_id)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class EntropyConv(nn.Module):

    def __init__(self, ngroups, cin, cout, hidden, npart, ctx:PseudoEntropyContext, device_id, act=True):
        super(EntropyConv,self).__init__()
        self.pad = PseudoEntropyPad(2,npart,ctx,device=device_id)
        self.conv = MaskConv2(ngroups,cin,cout,5,hidden,device_id)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
        self.act = nn.PReLU(ngroups*cout) if act else None

    def forward(self,x):
        tx = self.pad(x)
        tx = self.conv(tx)
        if self.act is not None: tx = self.act(tx)
        return self.trim(tx)


class EntropyResidualBlock(nn.Module):
    
    def __init__(self, ngroups, cpn, npart, ctx:PseudoEntropyContext, device_id=0):
        super(EntropyResidualBlock, self).__init__()
        self.conv1 = EntropyConv(ngroups,cpn,cpn,True,npart,ctx,device_id,True)
        self.conv2 = EntropyConv(ngroups,cpn,cpn,True,npart,ctx,device_id,True)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        return y+x

class EntropySubNet(nn.Module):

    def __init__(self, ngroups, cpn, npart, num_gaussian, net_type, ctx:PseudoEntropyContext, device_id):
        super(EntropySubNet,self).__init__()

        self.net = nn.Sequential(
            EntropyConv(ngroups,1,cpn,False,npart,ctx,device_id),
            EntropyResidualBlock(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlock(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlock(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlock(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlock(ngroups,cpn,npart,ctx,device_id),
            EntropyConv(ngroups,cpn,num_gaussian,True,npart,ctx,device_id,False)
        ) 

        self.reshape = ContextReshape(ngroups,device_id)
        self.act = None
        if net_type == 0:#weight
            self.act = nn.Softmax(dim=1)
        elif net_type == 2:#delta
            self.act = nn.ReLU() 
            self.net._modules['6'].conv.bias.data.fill_(2)

    def forward(self, x):
        tx = self.net(x)
        y = self.reshape(tx)
        if self.act is not None:
            y = self.act(y)
        return y


class EntropyNet(nn.Module):
    
    def __init__(self, ngroups, npart, ctx:PseudoEntropyContext, cpn=3, num_gaussian=3, device_id=0, drop_flag = False):
        super(EntropyNet,self).__init__()
        self.drop = DropGrad(drop_flag)
        self.weight_net = EntropySubNet(ngroups,cpn,npart,num_gaussian,0,ctx,device_id)
        self.mean_net = EntropySubNet(ngroups,cpn,npart,num_gaussian,1,ctx,device_id)
        self.delta_net = EntropySubNet(ngroups,cpn,npart,num_gaussian,2,ctx,device_id)
        self.mask = None
        self.fill = PseudoFillV2(0,npart,ctx,device=device_id)
        self.fill2 = PseudoFillV2(0,npart,ctx,device=device_id)
        self.ent_loss = EntropyGmm(num_gaussian,device_id)

    def setup_mask(self,x):
        with torch.no_grad():
            self.mask = torch.ones_like(x).detach()
            self.mask = self.fill(self.mask)
            self.mask = self.mask.view(-1)
        return

    def forward(self,x):
        self.setup_mask(x)
        x = self.fill2(x)
        tx = self.drop(x)
        weight = self.weight_net(tx)
        mean = self.mean_net(tx)
        delta = self.delta_net(tx) + 1e-6
        label = tx.view(-1,1)
        loss_vec = self.ent_loss(weight, delta, mean, label)
        return loss_vec*self.mask, self.mask


class CMPNetV2M(nn.Module):
    
    def __init__(self, valid_dim=162, channels=192, code_channels=192, npart=16, quant_levels=8, opt=False, init=False, device_id=0):
        super(CMPNetV2M, self).__init__()
        self.slice = SphereSlice(npart,pad=0,opt=opt,device=device_id)
        self.uslice = SphereUslice(npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(npart,opt,device=device_id)
        self.encoder = EncoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.decoder = DecoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.quant = PseudoQUANTV2(code_channels,quant_levels,npart, self.ctx, top_alpha=0.0001, device_id=device_id,ntop=1)#top_alpha mse:0.0001 ssim:0.01
        self.vm = StubMask(valid_dim)
        self.clip = ClipData()
        self.mean_val = (quant_levels - 1) / 2.
        self.dtw = Dtow(2, True, version=0, device=device_id)

    def forward(self,x):
        x = self.slice(x)
        code = self.encoder(x)
        code_f = self.quant(code)
        vmask = self.vm(code_f)
        code_f=code_f*vmask
        tx = self.decoder(code_f)
        tx = self.uslice(tx)
        return self.clip(tx)

class CMPNetV2MF(nn.Module):
    
    def __init__(self, valid_dim=162, channels=192, code_channels=192, npart=16, quant_levels=8, opt=False, init=False, device_id=0):
        super(CMPNetV2MF, self).__init__()
        self.slice = SphereSlice(npart,pad=0,opt=opt,device=device_id)
        self.uslice = SphereUslice(npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(npart,opt,device=device_id)
        self.ctx_ent = PseudoEntropyContext(npart,1,opt,device=device_id)
        self.encoder = EncoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.decoder = DecoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.quant = PseudoQUANTV2(code_channels,quant_levels,npart, self.ctx, check_iters=20000, top_alpha=0.0001, device_id=device_id,ntop=2)#top_alpha mse:0.0001 ssim:0.01
        self.vm = StubMask(valid_dim)
        self.ext = Extract(valid_dim)
        self.clip = ClipData()
        self.ent = EntropyNet(valid_dim//4,npart,self.ctx_ent,3,3,device_id,drop_flag=init)
        self.mean_val = (quant_levels - 1) / 2.
        self.dtw = Dtow(2, True, version=0, device=device_id)

    def forward(self,x):
        x = self.slice(x)
        code = self.encoder(x)
        code_f, code_i = self.quant(code)
        vmask = self.vm(code_f)
        code_f=code_f*vmask
        tx = self.decoder(code_f)
        tx = self.uslice(tx)
        code_i=self.ext(code_i)
        hcode_i = self.dtw(code_i)
        qy = hcode_i - self.mean_val
        ent_vec, mask = self.ent(qy)
        return self.clip(tx), ent_vec, mask

class CMPNetV2MFExtractor(nn.Module):
    
    def __init__(self, valid_dim=162, channels=192, code_channels=192, npart=16, quant_levels=8, opt=False, init=False, device_id=0):
        super(CMPNetV2MFExtractor, self).__init__()
        self.slice = SphereSlice(npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(npart,opt,device=device_id)
        self.encoder = EncoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.quant = PseudoQUANTV2(code_channels,quant_levels,npart, self.ctx, top_alpha=0.0001, device_id=device_id,ntop=2)#top_alpha mse:0.0001 ssim:0.01
        self.ext = Extract(valid_dim)
        self.mean_val = (quant_levels - 1) / 2.
        self.dtw = Dtow(2, True, version=0, device=device_id)

    def forward(self,x):
        x = self.slice(x)
        code = self.encoder(x)
        _, code_i = self.quant(code)
        code_i=self.ext(code_i)
        hcode_i = self.dtw(code_i)
        return hcode_i

class CMPNetV2Decoder(nn.Module):
    
    def __init__(self, channels=192, code_channels=192, npart=16, opt=False, init=False, device_id=0):
        super(CMPNetV2Decoder, self).__init__()
        self.uslice = SphereUslice(npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(npart,opt,device=device_id)
        self.decoder = DecoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.clip = ClipData()

    def forward(self,x):
        tx = self.decoder(x)
        tx = self.uslice(tx)
        return self.clip(tx)

class CMPNetV2MFEntropy(nn.Module):
    
    def __init__(self, valid_dim=162, channels=192, code_channels=192, npart=16, quant_levels=8, opt=False, init=False, device_id=0):
        super(CMPNetV2MFEntropy, self).__init__()
        self.ctx = PseudoEntropyContext(npart,1,opt,device=device_id)
        self.ent = EntropyNet(valid_dim//4,npart,self.ctx,3,3,device_id,drop_flag=init)
        self.mean_val = (quant_levels - 1) / 2.

    def forward(self,x):
        qy = x - self.mean_val
        ent_vec, mask = self.ent(qy)
        return ent_vec, mask

class AccGrad():

    def __init__(self, params):
        self.acc_grad = []
        for p in list(params):
            self.acc_grad.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        self.num_param = len(self.acc_grad)
        
    def zero(self):
        for idx in range(self.num_param):
            self.acc_grad[idx].zero_()

    def acc(self,params):
        grad_list = [p.grad for p in list(params)]
        torch._foreach_add_(self.acc_grad,grad_list)

    def copy_back(self,param):
        grad_list = [p.grad for p in list(param)]
        torch._foreach_add_(grad_list,self.acc_grad)
        self.zero()

class EntropyConv2(nn.Module):
    
    def __init__(self, ngroups, cin, cout, hidden, npart, ctx:PseudoContextV2, device_id, act=True):
        super(EntropyConv2,self).__init__()
        self.pad = PseudoPadV2(1,npart,ctx,device=device_id)
        self.conv = ChannelGroupConv(ngroups,cin,cout,3,hidden,device_id)
        self.trim = PseudoFillV2(0,npart,ctx,device=device_id)
        self.act = nn.PReLU(ngroups*cout) if act else None

    def forward(self,x):
        tx = self.pad(x)
        tx = self.conv(tx)
        if self.act is not None: tx = self.act(tx)
        return self.trim(tx)


class EntropyResidualBlockV2(nn.Module):
    
    def __init__(self, ngroups, cpn, npart, ctx:PseudoContextV2, device_id=0):
        super(EntropyResidualBlockV2, self).__init__()
        self.conv1 = EntropyConv2(ngroups,cpn,cpn,True,npart,ctx,device_id,True)
        self.conv2 = EntropyConv2(ngroups,cpn,cpn,True,npart,ctx,device_id,True)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        return y+x

class EntropySubNetV2(nn.Module):

    def __init__(self, code_channels, ngroups, cpn, npart, num_gaussian, net_type, ctx:PseudoContextV2, device_id):
        super(EntropySubNetV2,self).__init__()
        cin = code_channels // ngroups
        self.net = nn.Sequential(
            EntropyConv2(ngroups,cin,cpn,False,npart,ctx,device_id),
            EntropyResidualBlockV2(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlockV2(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlockV2(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlockV2(ngroups,cpn,npart,ctx,device_id),
            EntropyResidualBlockV2(ngroups,cpn,npart,ctx,device_id),
            EntropyConv2(ngroups,cpn,num_gaussian,True,npart,ctx,device_id,False)
        ) 

        self.reshape = ChannelGroupReshape(ngroups,code_channels,device_id)
        self.act = None
        if net_type == 0:#weight
            self.act = nn.Softmax(dim=1)
        elif net_type == 2:#delta
            self.act = nn.ReLU() 
            self.net._modules['6'].conv.bias.data.fill_(2)

    def forward(self, x):
        tx = self.net(x)
        y = self.reshape(tx)
        if self.act is not None:
            y = self.act(y)
        return y


class EntropyNetV2(nn.Module):
    
    def __init__(self, code_channels, ngroups, npart, ctx:PseudoContextV2, cpn=3, num_gaussian=3, device_id=0, drop_flag = False):
        super(EntropyNetV2,self).__init__()
        self.drop = DropGrad(drop_flag)
        self.weight_net = EntropySubNetV2(code_channels,ngroups,cpn,npart,num_gaussian,0,ctx,device_id)
        self.mean_net = EntropySubNetV2(code_channels,ngroups,cpn,npart,num_gaussian,1,ctx,device_id)
        self.delta_net = EntropySubNetV2(code_channels,ngroups,cpn,npart,num_gaussian,2,ctx,device_id)
        self.mask = None
        self.fill = PseudoFillV2(0,npart,ctx,device=device_id)
        self.fill2 = PseudoFillV2(0,npart,ctx,device=device_id)
        self.ent_loss = EntropyGmm(num_gaussian,device_id)

    def setup_mask(self,x):
        with torch.no_grad():
            self.mask = torch.ones_like(x).detach()
            self.mask = self.fill(self.mask)
            self.mask = self.mask.view(-1)
        return

    def forward(self,x):
        self.setup_mask(x)
        x = self.fill2(x)
        tx = self.drop(x)
        weight = self.weight_net(tx)
        mean = self.mean_net(tx)
        delta = self.delta_net(tx) + 1e-6
        label = tx.view(-1,1)
        loss_vec = self.ent_loss(weight, delta, mean, label)
        return loss_vec*self.mask, self.mask

class CMPNetV3MF(nn.Module):

    def __init__(self, valid_dim=162, channels=192, code_channels=192, npart=16, quant_levels=8, opt=False, init=False, device_id=0):
        super(CMPNetV3MF, self).__init__()
        self.slice = SphereSlice(npart,pad=0,opt=opt,device=device_id)
        self.uslice = SphereUslice(npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(npart,opt,device=device_id)
        self.encoder = EncoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.decoder = DecoderV2(channels,code_channels,npart,self.ctx,device_id)
        self.quant = PseudoQUANTV2(code_channels,quant_levels,npart, self.ctx, check_iters=20000, top_alpha=0.0001, device_id=device_id,ntop=2)#top_alpha mse:0.0001 ssim:0.01
        self.vm = StubMask(valid_dim)
        self.ext = Extract(valid_dim)
        self.clip = ClipData()
        #self.ent = EntropyNetV2(valid_dim,valid_dim//2,npart,self.ctx,3,3,device_id,drop_flag=init)
        self.ent = EntropyNetV2(valid_dim//4,valid_dim//4,npart,self.ctx,8,3,device_id,drop_flag=init)
        self.mean_val = (quant_levels - 1) / 2.
        self.wtd = Dtow(2, True, version=0, device=device_id)

    def forward(self,x):
        x = self.slice(x)
        code = self.encoder(x)
        code_f, code_i = self.quant(code)
        vmask = self.vm(code_f)
        code_f=code_f*vmask
        tx = self.decoder(code_f)
        tx = self.uslice(tx)
        code_i=self.ext(code_i)
        hcode_i = self.wtd(code_i)
        qy = hcode_i - self.mean_val
        ent_vec, mask = self.ent(qy)
        return self.clip(tx), ent_vec, mask
            
def test():
    net = CMPNetV2MF(168,192,192,16,8,False,0).to('cuda:0')
    for _ in range(3):
        data = torch.rand(1,3,512,1024).to('cuda:0')
        y,ent,mask = net(data)
        loss = torch.mean(y**2/2)+torch.sum(ent)/torch.sum(mask).item()
        #y = net(data)
        #loss = torch.mean(y**2/2)
        loss.backward()
    pass


if __name__ == '__main__':
    test()
    