from numpy.lib.function_base import average, interp
import torch
from torch import nn
from PCONV_operator import EntropyContextNew, EntropyConv2Batch, EntropyCtxPadRun2
from PCONV_operator import DExtract2, DInput2, PseudoFillV2, DExtract2Batch,MultiProject
from PCONV_operator import EntropyAdd, EntropyBatchGmmTable,PseudoContextV2, Dtow, PseudoQUANTV2,Extract, PseudoDQUANT
from PCONV_operator import SphereSlice, SphereUslice, SSIM
import coder
from model_zoo_v2 import EncoderV2, DecoderV2, ClipData
import os
import numpy as np
import cv2
from collections import OrderedDict
import argparse
import math
psnr_f = lambda xa: 10*math.log10(1./xa)

model_ssim_list = ['1_56', '2_56', '3_56', '4_56', '5_112', '6_112', '7_112', '8_192', '9_192']
ssim_channel_list=[56, 56, 56, 56, 112, 112, 112, 192, 192]
model_mse_list = ['1_56', '2_56', '3_56', '4_112', '5_112', '6_112', '7_112', '8_192', '9_192', '10_192']
mse_channel_list=[56, 56, 56, 112, 112, 112, 112, 192, 192, 192]
mse_model_dir = './demo/mse'
ssim_model_dir = './demo/ssim'



class EntropyConvDBT(nn.Module):
    
    def __init__(self, batch,ngroups, cin, cout, hidden, npart, out_layer:bool, ctx:EntropyContextNew, device_id, act=True):
        super(EntropyConvDBT,self).__init__()
        pad_out = 0 if out_layer else 2
        self.pad = EntropyCtxPadRun2(2,npart,ngroups,ctx,not hidden,device=device_id)#
        self.conv = EntropyConv2Batch(npart,ngroups,cin,cout,5,ctx,2,pad_out,batch=batch,hidden=hidden,act=act,device=device_id)

    def forward(self,x):
        tx = self.pad(x)
        tx = self.conv(tx)
        return tx

class EntropyResidualBlockDBT(nn.Module):
    
    def __init__(self, batch, ngroups, cpn, npart, ctx:EntropyContextNew, device_id=0):
        super(EntropyResidualBlockDBT, self).__init__()
        self.conv1 = EntropyConvDBT(batch,ngroups,cpn,cpn,True,npart,False,ctx,device_id,True)
        self.conv2 = EntropyConvDBT(batch,ngroups,cpn,cpn,True,npart,False,ctx,device_id,True)
        self.add = EntropyAdd(npart,cpn*ngroups,ngroups,2,ctx,device=device_id)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        y = self.add(y,x)
        return y

@torch.no_grad()
def restart_entropy_network(m):
    if isinstance(m,EntropyConv2Batch):
        m.restart()
    elif isinstance(m,EntropyCtxPadRun2):
        m.restart()
    elif isinstance(m,EntropyAdd):
        m.restart()
    elif isinstance(m,DInput2):
        m.restart()
    elif isinstance(m,DExtract2):
        m.restart()
    elif isinstance(m,DExtract2Batch):
        m.restart()

class EntEncoder(nn.Module):

    def __init__(self, ngroup, npart=16 ,opt_f=True, bin_num=8, gid=0):
        super(EntEncoder,self).__init__()
        self.cuda = 'cuda:{}'.format(gid)
        self.ctx2 = EntropyContextNew(npart,opt=opt_f,device=gid)
        self.ipt = DInput2(ngroup,npart,self.ctx2,2,-3.5,3,device=gid)
        self.npart,self.ngroup = npart,ngroup
        self.fill = PseudoFillV2(0,npart,self.ctx2,0,device=gid)
        self.mcoder = None#coder.coder(code_name)
        self.bias = (bin_num-1)/2.
        self.net = nn.Sequential(
            EntropyConvDBT(3,ngroup,1,3,False,npart,False,self.ctx2,gid,True),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyConvDBT(3,ngroup,3,3,True,npart,True,self.ctx2,gid,False)
        )
        self.ext = DExtract2Batch(npart,ngroup,self.ctx2,device=gid)
        self.ext_label = DExtract2(npart,ngroup,True,self.ctx2,device=gid)
        self.gmm = EntropyBatchGmmTable(bin_num,self.bias,3,65536,device=gid)
        self.net=self.net.to(self.cuda)
        
    def start(self, code_name='./tmp/data'):
        self.apply(restart_entropy_network)
        self.mcoder = coder.coder(code_name)

    def forward(self,data):
        with torch.no_grad():
            data = self.fill(data)
            h,w = data.shape[2:]
            self.ctx2.setup_context(w)
            self.mcoder.start_encoder()
            h_full = h*self.npart
            label = torch.zeros((1,1,h_full,w),dtype=torch.float32).to(self.cuda)
            for _ in range(h_full+w+self.ngroup-2):
                b = self.ipt(label)
                y = self.net(b)
                z,le = self.ext(y)
                vec = self.gmm(z,le)
                ln = int(le[0].item())
                label,_ = self.ext_label(data)
                pred, tlabel = vec.type(torch.int32).to('cpu'), label.type(torch.int32).to('cpu')
                self.mcoder.encodes(pred,8,tlabel,ln)
            self.mcoder.end_encoder()


class EntDecoder(nn.Module):
    
    def __init__(self, ngroup, npart=16 ,opt_f=True, bin_num=8, gid=0):
        super(EntDecoder,self).__init__()
        self.cuda = 'cuda:{}'.format(gid)
        self.ctx2 = EntropyContextNew(npart,opt=opt_f,device=gid)
        self.ipt = DInput2(ngroup,npart,self.ctx2,2,-3.5,3,device=gid)
        self.npart,self.ngroup = npart,ngroup
        self.mcoder = None#coder.coder(code_name)
        self.bias = (bin_num - 1)/2.
        self.fill = PseudoFillV2(0,npart,self.ctx2,0,device=gid)
        self.net = nn.Sequential(
            EntropyConvDBT(3,ngroup,1,3,False,npart,False,self.ctx2,gid,True),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyResidualBlockDBT(3,ngroup,3,npart,self.ctx2,gid),
            EntropyConvDBT(3,ngroup,3,3,True,npart,True,self.ctx2,gid,False)
        )
        self.ext = DExtract2Batch(npart,ngroup,self.ctx2,device=gid)
        self.gmm = EntropyBatchGmmTable(bin_num,self.bias,3,65536,device=gid)
        self.net=self.net.to(self.cuda)
        
    def start(self, code_name='./tmp/data'):
        self.apply(restart_entropy_network)
        self.mcoder = coder.coder(code_name)

    def forward(self,h,w):
        with torch.no_grad():
            self.ctx2.setup_context(w)
            self.mcoder.start_decoder()
            h_full = h*self.npart
            pout = torch.zeros((1,1,h_full,w),dtype=torch.float32).to(self.cuda)
            for _ in range(h_full+w+self.ngroup-2):
                b = self.ipt(pout) 
                y = self.net(b)
                z,le = self.ext(y)
                vec = self.gmm(z,le)
                ln = int(le[0].item())
                pred = vec.type(torch.int32).to('cpu').view(-1,9)
                pout = self.mcoder.decodes(pred,8,ln).to(self.cuda).view(1,1,h_full,w).contiguous()
            code = (b[:self.npart,:,2:-2,2:-2] + self.bias).contiguous()
            return self.fill(code)

class PseudoEncoder(nn.Module):

    def __init__(self, valid_dim, device_id):
        super(PseudoEncoder,self).__init__()
        npart,opt,channels,code_channels=16,True,192,192
        quant_levels = 8
        #valid_dim = 192
        self.slice = SphereSlice(npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(npart,opt,device=device_id)
        self.encoder = EncoderV2(channels,code_channels,npart,self.ctx,device_id).to('cuda:{}'.format(device_id))
        self.quant = PseudoQUANTV2(code_channels,8,npart, self.ctx, device_id=device_id,ntop=2)
        self.ext = Extract(valid_dim)
        self.mean_val = (quant_levels - 1) / 2.
        self.dtw = Dtow(2, True, device_id)
        self.ent = EntEncoder(valid_dim//4,npart,opt,quant_levels,gid=device_id)

    def forward(self,x,code_name):
        with torch.no_grad():
            x = self.slice(x)
            code = self.encoder(x)
            _, code_i = self.quant(code)
            code_i=self.ext(code_i)
            hcode_i = self.dtw(code_i)
            self.ent.start(code_name)
            self.ent(hcode_i)

class PseudoDecoder(nn.Module):
    
    def __init__(self, valid_dim, device_id):
        super(PseudoDecoder,self).__init__()
        self.npart,opt,self.channels,self.code_channels=16,True,192,192
        quant_levels = 8
        self.valid_dim = valid_dim#192
        self.uslice = SphereUslice(self.npart,pad=0,opt=opt,device=device_id)
        self.ctx = PseudoContextV2(self.npart,opt,device=device_id)
        self.decoder = DecoderV2(self.channels,self.code_channels,self.npart,self.ctx,device_id).to('cuda:{}'.format(device_id))
        self.clip = ClipData()
        self.quant = PseudoDQUANT(self.code_channels,8,self.npart, self.ctx, device_id=device_id)
        self.wtd = Dtow(2, False, device_id)
        self.ent = EntDecoder(self.valid_dim//4,self.npart,opt,quant_levels,gid=device_id)

    def forward(self,code_name):
        with torch.no_grad():
            self.ent.start(code_name)
            hcode_i = self.ent(4,128)
            code_i = self.wtd(hcode_i)
            code_ext = self.quant(code_i)
            code_f = torch.zeros((self.npart,self.code_channels,2,64)).type_as(code_ext)
            code_f[:,:self.valid_dim] = code_ext
            tx = self.decoder(code_f.contiguous())
            tx = self.uslice(tx)
            return self.clip(tx)

def img2tensor(img,device):
    ts = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))/255.
    return torch.unsqueeze(ts,0).to(device).contiguous()

def tensor2img(data):
    img = (data[0]*255.).to('cpu').detach().numpy().transpose(1,2,0)
    return img.astype(np.uint8)

def load_models(model:nn.Module,p1,p2,device):
    d2 = torch.load(p1,device)
    d1 = torch.load(p2,device)
    md = OrderedDict(**d2,**d1)
    model.load_state_dict(md)

def check_img(img):
    h,w = img.shape[:2]
    if not(h==512 and w==1024):
        return cv2.resize(img,(1024,512),interp=cv2.INTER_CUBIC)
    else:
        return img

def encoding(img_list, out_list, model_idx=0, mse=True, device_id = 0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    vd = mse_channel_list[model_idx] if mse else ssim_channel_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    cuda = 'cuda:{}'.format(device_id)
    t1 = PseudoEncoder(vd,device_id=device_id).to(cuda)
    load_models(t1,'{}/{}_encoder.pt'.format(model_dir,prex),'{}/{}_ent.pt'.format(model_dir,prex),cuda)
    for fn, fo in zip(img_list,out_list):
        img = check_img(cv2.imread(fn))
        data = img2tensor(img,cuda)
        t1(data,fo)
        print('Encoding {}, bitrate: {:.3f}bpp'.format(fn,os.path.getsize(fo)*8/1024./512.))

def decoding(code_list, decoded_img_list, model_idx=0,mse=True, device_id=0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    vd = mse_channel_list[model_idx] if mse else ssim_channel_list[model_idx]
    cuda = 'cuda:{}'.format(device_id)
    t1 = PseudoDecoder(vd,device_id=device_id).to(cuda)
    load_models(t1,'{}/{}_decoder.pt'.format(model_dir,prex),'{}/{}_ent.pt'.format(model_dir,prex),cuda)
    for fc,fo in zip(code_list,decoded_img_list):
        rdata = t1(fc)
        img = tensor2img(rdata)
        cv2.imwrite(fo,img)
        print('Decoding {}, output to {}'.format(fc,fo))
    

def decoding_and_test(code_list, img_list, model_idx=0,mse=True,device_id=0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    vd = mse_channel_list[model_idx] if mse else ssim_channel_list[model_idx]
    cuda = 'cuda:{}'.format(device_id)
    t1 = PseudoDecoder(vd,device_id=device_id).to(cuda)
    load_models(t1,'{}/{}_decoder.pt'.format(model_dir,prex),'{}/{}_ent.pt'.format(model_dir,prex),cuda)
    pr1 = MultiProject(171, int(171*1.5), 0.5, False, 0).to(cuda)
    pr2 = MultiProject(171, int(171*1.5), 0.5, False, 0).to(cuda)
    sim_func = SSIM(11, 3).to(cuda)
    rt_list, pr_list, ssim_list = [], [], []
    for fc, fn in zip(code_list,img_list):
        rdata = t1(fc)
        img = check_img(cv2.imread(fn))
        data = img2tensor(img,cuda)
        x = pr1(data)
        y = pr2(rdata)
        mse_loss = torch.mean((x-y)**2).item()
        pr = psnr_f(mse_loss)
        vssim = sim_func(x,y).item()
        rt = os.path.getsize(fc)*8/1024./512.
        rt_list.append(rt)
        pr_list.append(pr)
        ssim_list.append(vssim)
        print('Decoding {}, compare it to {} \n Bitrate:{:.3f}bpp, PSNR:{:.2f}dB, SSIM:{:.4f}'.format(fc, fn, rt, pr, vssim))
    print('-----------------------------------------------------\nAverage Performance\n-----------------------------------------------------')
    rt,pr,vssim = np.average(np.array(rt_list)), np.average(np.array(pr_list)), np.average(np.array(ssim_list))
    print('Bitrate:{:.3f}bpp, PSNR:{:.2f}dB, SSIM:{:.4f}'.format(rt, pr, vssim))

def test_pseudo_image_coding():
    img_dir = 'e:/360_dataset/360_512'
    code_dir = './tmp'
    with open('e:/360_dataset/test.txt') as f:
        test_list = [pt[:-1] for pt in f.readlines()]
    img_list = ['{}/{}'.format(img_dir,fn) for fn in test_list[:10]]
    code_list = ['{}/{}'.format(code_dir,idx) for idx in range(10)]
    encoding(img_list,code_list,9)
    decoding_and_test(code_list,img_list,9)

def read_list(fname):
    with open(fname) as f:
        return [line.rstrip('\n') for line in f.readlines()]

def check_models():
    assert(os.path.exists('{}/{}_encoder.pt'.format(mse_model_dir,model_mse_list[0]))),'Please make sure the pretrained models for VMSE exists in the mse_model_dir'
    assert(os.path.exists('{}/{}_encoder.pt'.format(ssim_model_dir,model_ssim_list[0]))),'Please make sure the pretrained models for VSSIM exists in the ssim_model_dir'

if __name__ == '__main__':
    #test_pseudo_image_coding()
    parser = argparse.ArgumentParser(description='Pseudo Convolution for 360 Image Compression')
    parser.add_argument('--img-list', nargs='*', help='The image list contains the input images for encoding and testing')
    parser.add_argument('--code-list', nargs='*', help='The code file list for codes')
    parser.add_argument('--out-list', nargs='*', help='The out list for saving decoded images.')
    parser.add_argument('--img-file', help='The file contains the input images for encoding and testing')
    parser.add_argument('--code-file', help='The file contains the list for codes')
    parser.add_argument('--out-file', help='The file  contains the names of decoded images.')
    parser.add_argument('--model-idx', type=int, default=0, help='Model index (0-9) for VMSE, (0-8) for VSSIM')
    parser.add_argument('--enc', action='store_true', default=False, help='Encoding flag, set for encoding phase.')
    parser.add_argument('--dec', action='store_true', default=False, help='Decoding flag, set for decoding phase.')
    parser.add_argument('--test', action='store_true', default=False, help='Testing flag, set for decoding and evalating the performance.')
    parser.add_argument('--ssim', action='store_true', default=False, help='Default with models optimized for VMSE, \
        set this flag for choosing the models optimized for VSSIM')
    parser.add_argument('--gpu-id', type=int, default=0, help='The graphic card id for encoding and decoding.')
    args = parser.parse_args()
    check_models()
    midx = args.model_idx
    if args.ssim:
        assert(midx<9 and midx>=0),'(0-8) for VSSIM'
    else:
        assert(midx<10 and midx>=0),'(0-9) for VMSE'
    assert(args.enc or args.dec or args.test),'Should set one flag, (--enc) for encoding, (--dec) for decoding, (--test) for testing.'
    img_lnone, img_fnone = args.img_list is not None, args.img_file is not None
    code_lnone, code_fnone = args.code_list is not None, args.code_file is not None
    out_lnone, out_fnone = args.out_list is not None, args.out_file is not None
    if args.enc:
        assert(img_fnone or img_lnone), 'No input images for encoding'
        assert(code_lnone or code_fnone), 'No code files for saving the codes'
        img_list = args.img_list if img_lnone else read_list(args.img_file)
        code_list = args.code_list if code_lnone else read_list(args.code_file)
        assert(len(img_list)==len(code_list)), 'The number of images and codes should be the same'
        encoding(img_list,code_list,midx,not args.ssim,args.gpu_id)
    else:
        assert(code_lnone or code_fnone), 'No code files for decoding'
        code_list = args.code_list if code_lnone else read_list(args.code_file)
        if args.dec:
            assert(out_lnone or out_fnone), 'No out files for saving the decoded images'
            out_list = args.out_list if out_lnone else read_list(args.out_file)
            assert(len(code_list)==len(out_list)), 'The number of codes and reconstructed images should be the same'
            decoding(code_list,out_list,midx,not args.ssim,args.gpu_id)
        else:
            assert(img_fnone or img_lnone), 'No source images for evaluation.'
            img_list = args.img_list if img_lnone else read_list(args.img_file)
            assert(len(code_list)==len(img_list)), 'The number of codes and corresponding source images should be the same'
            decoding_and_test(code_list,img_list,midx,not args.ssim,args.gpu_id)
