import numpy as np
import scipy.interpolate
import os

def load_param(fileName):
    if os.path.exists(fileName):
        f = open(fileName)
        param = [int(pt) for pt in f.readline()[:-1].split(',')]
    else:
        param = [8,18,24,36,46,58,62,62,62,62,63,63,63,63,63,63,63,63,63,63,63,63,62,62,62,62,58,46,36,24,18,8]
    return param

def set_weight(npart,opt=False,merge=False,config_file='./config/param.txt'):
    assert npart % 2 == 0, 'npart should be the multiplier of 2 for the merge case'
    tnpart = npart * 2 if merge else npart
    if opt:
        #vlist=[8,18,24,36,46,58,62,62,62,62,63,63,63,63,63,63,63,63,63,63,63,63,62,62,62,62,58,46,36,24,18,8]
        vlist = load_param(config_file)
        y = np.array([pa+1 for pa in vlist])
        #cos(((tidx[i] - 0.5)/height - 0.5)*pi)+0.5)
        pi = np.pi
        x = np.cos((0.5- (np.arange(32.) + 0.5) / 32 )*pi)
        xt = np.cos((0.5- (np.arange(tnpart) + 0.5) / tnpart )*pi)
        hp = tnpart//2
        yt_a = np.ceil(scipy.interpolate.pchip_interpolate(x[:16],y[:16],xt[:hp]))
        yt_b = np.ceil(scipy.interpolate.pchip_interpolate(x[16:][::-1],y[16:][::-1],xt[hp:]))
        rlist = yt_a.tolist() + yt_b.tolist()
        if merge:  rlist = [max(rlist[2*idx], rlist[2*idx+1]) for idx in range(tnpart//2)]
        return rlist
    else:
        rt = 1.
        ya = np.ceil(np.cos((0.5- (np.arange(tnpart) + 0.5) / tnpart )*np.pi)*64.*rt + 64.*(1-rt))
        rlist = ya.tolist()
        if merge:  rlist = [max(rlist[2*idx], rlist[2*idx+1]) for idx in range(tnpart//2)]
        return rlist



if __name__ == '__main__':
    print(set_weight(16))
    print(set_weight(16,True))
    print(set_weight(8,True,True))