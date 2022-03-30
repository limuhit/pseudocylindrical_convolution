from __future__ import print_function
import os
import argparse
import torch
import time,random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from PCONV_operator import ModuleSaver, Logger, MultiProject, SSIM
from SphereDataset import load_train_test_distribute 
import model_zoo_v2
from model_zoo_v2 import AccGrad
from itertools import chain
from RDMetric import mse_tb,ssim_tb
base_dir = '/data1/home/csmuli/PConv'

def get_params(model):
    param = chain(model.module.encoder.parameters(), model.module.decoder.parameters(), [model.module.quant.weight])
    return param

def train(args, model, device, train_loader, optimizer, optimizer_quant, epoch, log, pr1, pr2, ent=True):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    param = get_params(model)
    acc_grad = AccGrad(param)
    sim_func = SSIM(11, 3).to(device)
    gamma,beta,clip = args.gamma, args.beta, args.clip
    log.log('clip:{}'.format(clip))
    acc_batch = args.acc_batch
    acc_batch_m1 = acc_batch -1
    for batch_idx, data in enumerate(train_loader):
        if not data.shape[0]==args.batch_size:continue
        data = data.to(device)
        optimizer.zero_grad()
        optimizer_quant.zero_grad()
        y = model(data)
        py = pr1(y)
        px = pr2(data)
        mse_loss = torch.mean((px-py)*(px-py)) 
        ssim_loss = 1-sim_func(px,py)
        loss = gamma*mse_loss + beta*ssim_loss
        loss.backward()
        optimizer_quant.step()
        param = get_params(model)
        if batch_idx % acc_batch == acc_batch_m1:
            acc_grad.copy_back(param)       
            torch.nn.utils.clip_grad_norm_(param,clip)
            optimizer.step()
        else:
            acc_grad.acc(param)
        log.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} mse:{:.6f} ssim:{:.3}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item(),mse_loss.item(), 1-ssim_loss.item()))

def test(args, model, device, test_loader,log, pr1, pr2):
    model.eval()
    sim_func = SSIM(11, 3).to(device)
    test_mse,test_ssim,test_loss = 0, 0, 0
    gamma,beta = args.gamma, args.beta
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)
            y = model(data)
            py = pr1(y)
            px = pr2(data)
            mse_loss = torch.mean((px-py)*(px-py)) 
            ssim_loss = 1-sim_func(px,py)
            loss = gamma*mse_loss + beta*ssim_loss
            test_mse += mse_loss.item()  # sum up batch loss
            test_ssim += (1-ssim_loss.item())
            test_loss += loss.item()
    test_mse /= len(test_loader)
    test_ssim /= len(test_loader)
    test_loss /= len(test_loader)
    log.log('\nTest set: MSE loss: {:.6f}  ssim loss: {:.4f}'.format(test_mse, test_ssim))
    rt_loss = [test_loss]
    loss_str = 'tloss: '+ '{}\t'*len(rt_loss)
    log.log(loss_str.format(*rt_loss))
    return rt_loss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)

def init_with_trained_model(path, model, device):
    pdict = torch.load(path,map_location=device)
    ndict = model.state_dict()
    for pkey in ndict.keys():
        if pkey in pdict.keys():
            ndict[pkey] = pdict[pkey]
    model.load_state_dict(ndict)

def Job(rank, world_size, args):
    cid = args.gpu_ids[rank]
    args.gpu_id = cid
    time.sleep(random.random()*10)
    torch.manual_seed(int(time.time()))
    setup(rank,world_size)
    device = torch.device("cuda:%d"%cid)
    train_loader,test_loader = load_train_test_distribute(world_size,rank, args.batch_size, args.test_batch_size, mean = 1.5, acc_batch=args.acc_batch)
    if not os.path.exists('{}/save_models/'.format(base_dir)) and rank==0: os.mkdir('{}/save_models/'.format(base_dir))
    prex = 'base_{}_{}_{}_{}'.format('opt' if args.opt else 'normal', args.channels, args.valid_dim, args.npart) 
    prex = '{}_init'.format(prex) if args.init else prex
    log = Logger('{}/save_models/{:s}_logs_{}.txt'.format(base_dir,prex,cid),screen=False,file=(rank==0))
    viewport_size = args.viewport_size
    pr1 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    pr2 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    model = model_zoo_v2.CMPNetV2M(args.valid_dim, args.channels,args.code_dim,args.npart,opt=args.opt,init=args.init,device_id=cid)
    saver = ModuleSaver('{}/save_models/'.format(base_dir),prex) if rank == 0 else None

    if args.init:
        of = '{}/save_models/mse_192.pt'.format(base_dir)
        init_with_trained_model(of, model,device)
        model.to(device)
        model = DDP(model,[cid])
        log.log('load init model {} successful...'.format(of))
    elif os.path.exists('{}/save_models/{}_best_0.pt'.format(base_dir,prex)):
        of = '{}/save_models/{}_latest.pt'.format(base_dir,prex) if args.latest else '{}/save_models/{}_best_0.pt'.format(base_dir,prex)
        init_with_trained_model(of,model,device)
        model.to(device)
        model = DDP(model,[cid])
        ls = test(args, model, device, test_loader,log, pr1, pr2)
        if args.restart: ls = [1e9 for _ in range(len(ls))] if  isinstance(ls,list) else 1e9
        if rank==0: saver.init_loss(ls)
        log.log('load model successful...')
    else:
        of = '{}/save_models/{}_init_best_0.pt'.format(base_dir,prex)
        init_with_trained_model(of,model,device)
        model.to(device)
        model = DDP(model,[cid])
        log.log('initialize the model with {}...'.format(of))

    optimizer_quant = torch.optim.SGD([model.module.quant.count], lr=0.001)
    optimizer_other = torch.optim.Adam([{'params':model.module.encoder.parameters()},
                                {'params':model.module.decoder.parameters()}, 
                                {'params':[model.module.quant.weight]}], lr=args.lr)
    log.log('lr:{}'.format(args.lr))
    log.log('valid dims:{} \t '.format(args.valid_dim))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer_other, optimizer_quant, epoch, log, pr1, pr2,False)
        ls = test(args, model, device, test_loader, log, pr1, pr2)
        if rank == 0:  
            message = saver.save(model,ls)
            log.log(message)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 360 Compression')
    parser.add_argument('--gpu-ids', nargs='*', default=[0,3], metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--acc-batch', type=int, default=3)
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--valid-dim', type=int, default=192)
    parser.add_argument('--gamma', type=float, default=1, help='trade-off of MSE loss')
    parser.add_argument('--beta', type=float, default=0, help='trade-off of SSIM loss')
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, 
                        help='For Saving the current Model')
    parser.add_argument('--opt', action='store_true', default=True, 
                        help='Using opt spliting.')      
    parser.add_argument('--init', action='store_true', default=True, 
                        help='Using opt spliting.') 
    parser.add_argument('--latest', action='store_true', default=False, 
                        help='Using opt spliting.') 
    parser.add_argument('--restart', action='store_true', default=False) 
    parser.add_argument('--viewport_size', type=int, default = 171, metavar='viewport', 
                        help='viewport size for 360 projection.')
    parser.add_argument('--channels', type=int, default=192)
    parser.add_argument('--code-dim', type=int, default=192)
    parser.add_argument('--npart', type=int, default=16)
    parser.add_argument('--gpu-id', type=int, default=0, metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--version', type=int, default=0)
    args = parser.parse_args()
    args.gpu_ids = [int(pt) for pt in args.gpu_ids]
    world_size = len(args.gpu_ids)
    mp.spawn(Job,
             args=(world_size,args,),
             nprocs=world_size,
             join=True)
    
    
if __name__ == '__main__':
    main()
         
