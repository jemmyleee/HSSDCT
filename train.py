import os, argparse, glob, time, cv2, pdb, torch, models
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random as rn
from scipy.io import savemat, loadmat
from utils import *
from dataset import *
from math import acos, degrees
from tensorboardX import SummaryWriter 
from trainOps import *
from tqdm import tqdm

from models.hssdct import HyDCFN

torch.backends.cudnn.benchmark=True


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=6)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--resume_ind', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default="")
    parser.add_argument('--snr', type=int, default=35)
    
    parser.add_argument('--lr', type=float, default=0.000055)
    parser.add_argument('--step_size', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--finetuning_step', type=int, default=200, help='Works only if the mixed_align_opt is on')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate, 0 means training without weight decay')
    
    
    ## Data generator configuration
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bands', type=int, default=172)
    parser.add_argument('--msi_bands', type=int, default=4)
    parser.add_argument('--hsi_bands', type=int, default=172)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    parser.add_argument('--joint_loss', type=int, default=1)
    parser.add_argument('--gc', type=int, default=32)
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=172, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    parser.add_argument('--out_nc', type=int, default=172, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    parser.add_argument('--nf', type=int, default=96, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    parser.add_argument("--root", type=str, default="/ssd4t/Fusion_data", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="./data_path/val.txt")   
    parser.add_argument("--train_file", type=str, default="./data_path/train.txt")   
    parser.add_argument("--prefix", type=str, default="ASTUDY_num2_BAND4_SWINY_SNR0")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args


def trainer(args):
    # Print configuration
    print("#"*80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#"*80)
    
    flist = loadTxt(args.train_file)
    valfn = loadTxt(args.val_file)
    tlen = len(flist)
    print(f'#training samples is {tlen} and validation samples is {len(valfn)}')

    if args.network_mode==2:
        dataset = dataset_joint2
        print('Use triplet dataset')
    elif args.network_mode==1:
        dataset = dataset_joint
        print('Use pairwise (LRHSI+HRMSI) dataset')
    elif args.network_mode==0:
        dataset = dataset_h5
        print('Use CO dataset')
    
    train_loader = torch.utils.data.DataLoader(dataset(flist, args), batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(dataset(valfn, args, mode='val'), batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.workers)

    model = HyDCFN(args).to(args.device)
    if args.gpus>1:
        model = torch.nn.DataParallel(model).to(args.device)
    
    
    if args.resume_ind>0 or os.path.isfile(args.resume_ckpt):
        if not os.path.isfile(args.resume_ckpt):
            args.resume_ckpt = os.path.join('checkpoint', args.prefix, 'best.pth')
        if not os.path.isfile(args.resume_ckpt):
            print(f"checkpoint is not found at {args.resume_ckpt}")
            raise 
        state_dict = torch.load(args.resume_ckpt)  
        model.load_state_dict(state_dict)
        print(f'Loading the pretrained model from {args.resume_ckpt}')
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    if args.lr_scheduler=='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.resume_ind, eta_min=1e-10, last_epoch=-1)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
        
    L1Loss = torch.nn.SmoothL1Loss()

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(f'checkpoint/{args.prefix}'):
        os.mkdir(f'checkpoint/{args.prefix}')
    if not os.path.isdir('Rec'):
        os.mkdir('Rec')    
    if not os.path.isdir(f'Rec/{args.prefix}'):
        os.mkdir(f'Rec/{args.prefix}')
    
    writer = SummaryWriter('log/%s_exp2' % (args.prefix))
    
    resume_ind = args.resume_ind if args.resume_ind>0 else 0
    step = resume_ind
    best_sam = 99
    for epoch in range(resume_ind, args.epochs): 
            
        ep_loss = 0.
        running_loss, running_sam, running_bws=[],[],[]
        t1 = time.perf_counter()
        optim_time = time.perf_counter()
        for batch_idx, (X) in tqdm(enumerate(train_loader), total=len(train_loader)):
            t2 = time.perf_counter()-t1
            if args.DEBUG:
                print(f'Sampling time: {t2} seconds')
            
            t1 = time.perf_counter()
            if args.network_mode==2:
                x,x2,x3,y,_,_,_ = X
                x3 = x3.cuda()
                x2 = x2.cuda()
            elif args.network_mode==1:
                x,x2,y,_,_,_ = X
                x2 = x2.cuda()
            elif args.network_mode==0:
                x,y,_,_,_ = X
                
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            if args.DEBUG:
                print(f'To cuda tensor time {time.perf_counter()-t1} seconds')
            
            t1 = time.perf_counter()
            
            if args.network_mode==2:
                decoded = model(x, LRHSI=x2, HRMSI=x3)
            elif args.network_mode==1:
                decoded = model(LRHSI=x2, HRMSI=x)
            elif args.network_mode==0:
                decoded = model(x, LRHSI=None, HRMSI=None)
                
            if args.DEBUG:
                print(f'model inference time{time.perf_counter()-t1} seconds')
            loss = L1Loss(decoded, y)
            loss2 = sam_loss(decoded, y)
            loss3 = BandWiseMSE(decoded, y)
            
            while torch.isnan(loss2) and scheduler.get_last_lr()[0]>1e-12:
                print('Force learning rate decay to', scheduler.get_last_lr()[0]/5)
                
                args.resume_ckpt = os.path.join('checkpoint', args.prefix, 'last.pth')
                state_dict = torch.load(args.resume_ckpt)  
                model.load_state_dict(state_dict)
                
                optimizer = optim.Adam(model.parameters(), lr=scheduler.get_last_lr()[0]/5)  
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
                args.joint_loss = 0
                
                continue
                
            if torch.isnan(loss2):
                print('It is unnecessary to optimize anymore...abort the process')
                raise
                            
            reg = torch.std(decoded)
            if args.joint_loss==1:
                total_loss = loss + 0.1*loss2 + 0.1*loss3
            else:
                total_loss = loss

            t1 = time.perf_counter()
            total_loss.backward()
        
            if args.DEBUG:
                print(f'Backward time {time.perf_counter()-t1} seconds')
            
            optimizer.step()
            running_loss.append(loss.item())
            running_sam.append(loss2.item())
            running_bws.append(loss3.item())
            
        scheduler.step()
        optim_time = time.perf_counter()-optim_time

        if epoch% args.eval_step ==0:
            model.eval()
            with torch.no_grad():
                rmses, sams, fnames, psnrs, ergas = [], [], [], [], []
                
                ep = 0
                
                for ind2, X in tqdm(enumerate(val_loader), total=len(val_loader)):
                    if args.network_mode==2:
                        (vx, vx2, vx3, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                        vx3=vx3.cuda()
                    elif args.network_mode==1:
                        (vx, vx2, vy, vfn, maxv, minv) = X
                        vx2=vx2.cuda()
                    elif args.network_mode==0:
                        (vx, vy, vfn, maxv, minv) = X
                        
                    vy = vy.cpu().numpy()
                    vy = vy[:,:,:args.image_size,:args.image_size]
                    vx=vx.cuda()
                    
                    maxv, minv = maxv.cpu().numpy(), minv.cpu().numpy()
                   
                    start_time = time.time()
                    if args.network_mode==2:
                        val_dec = model(vx, LRHSI=vx2, HRMSI=vx3, mode=1)
                    elif args.network_mode==1:
                        val_dec = model(LRHSI=vx2, HRMSI=vx, mode=1)
                    elif args.network_mode==0:
                        val_dec = model(vx, LRHSI=None, HRMSI=None, mode=1)
                    ep = ep+(time.time()-start_time)
                    
                    val_dec = val_dec.cpu().numpy()
                    
                    
                    for predimg, gtimg,f, v1, v2 in zip(val_dec, vy, vfn, maxv, minv):
                        predimg = (predimg/2+0.5) 
                        gtimg = (gtimg/2+0.5) 
                        
                        sams.append(sam2(predimg, gtimg))
                        psnrs.append(psnr(predimg, gtimg))
                        ergas.append(ERGAS(predimg, gtimg))
                        predimg = predimg * (v1-v2) + v2
                        gtimg = gtimg * (v1-v2) + v2
                        rmses.append(rmse(predimg, gtimg))
                        savemat(f'Rec/{args.prefix}/{os.path.basename(f)}.mat', {'pred':np.transpose(predimg,(1,2,0))})
                                        
                ep = ep / len(sams)
                print('[epoch: %d] Loss: %.3f, Loss-SAM: %.3f, Loss-BWS: %.3f, val-rmse: %.3f, val-SAM: %.3f, val-PSNR: %.3f, val-ERGAS: %.3f, Inference time: %f ms, Optim time: %f, lr: %f' % (epoch, 100*np.mean(running_loss), np.mean(running_sam), np.mean(running_bws), np.mean(rmses), np.mean(sams), np.mean(psnrs), np.mean(ergas), ep*1000, optim_time,scheduler.get_last_lr()[0]))
                # Log validation metrics to TensorBoard
                writer.add_scalar('Validation RMSE', np.mean(rmses), step)
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                writer.add_scalar('Validation PSNR', np.mean(psnrs), step)
                writer.add_scalar('Validation ERGAS', np.mean(ergas), step)
                writer.add_scalar('Validation RMSE/std', np.std(rmses), step)
                writer.add_scalar('Validation SAM/std', np.std(sams), step)
                writer.add_scalar('Validation PSNR/std', np.std(psnrs), step)
                writer.add_scalar('Validation ERGAS/std', np.std(ergas), step)

            model.train() 
            
            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(),  f'checkpoint/{args.prefix}/best.pth')
                
            ep_loss += np.mean(running_loss)
            writer.add_scalar('Loss/Running loss', np.mean(running_loss), step)
            writer.add_scalar('Loss/Running SAM-loss', np.mean(running_sam), step)
            writer.add_scalar('Loss/Running Weighted-MSE', np.mean(running_bws), step)
                
            running_loss, running_sam, running_bws=[],[], []
            model.train() 
        
                
        torch.save(model.state_dict(), f'checkpoint/{args.prefix}/last.pth')
        
        step+=1


if __name__ == '__main__':
    
    args = parse_args()
    torch.manual_seed(args.SEED)
    rn.seed(args.SEED)
    np.random.seed(args.SEED)
    trainer(args)
