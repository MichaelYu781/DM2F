# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from model import DM2FNet_woPhy
from tools.config import OHAZE_ROOT
from datasets import OHazeDataset
from tools.utils import AvgMeter, check_mkdir, sliding_forward

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from torch.utils.tensorboard import SummaryWriter
from lpips import LPIPS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='1', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='bli',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 20000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 2e-4,
    'lr_decay': 0.9,
    'weight_decay': 2e-5,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 2000,
    'crop_size': 512,
}


def main():
    net = DM2FNet_woPhy().cuda().train()
    # net = DataParallel(net)

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, optimizer)

from loss import ContrastLoss

def train(net, optimizer):
    curr_iter = cfgs['last_iter']
    scaler = amp.GradScaler()
    torch.cuda.empty_cache()

    cts = ContrastLoss()

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record = AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()

        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']

            haze, gt, _ = data

            batch_size = haze.size(0)

            haze, gt = haze.cuda(), gt.cuda()

            optimizer.zero_grad()

            with amp.autocast():
                # import pdb; pdb.set_trace()
                x_jf, x_j1, x_j2, x_j3, x_j4 = net(haze)

                loss_x_jf = criterion(x_jf, gt)
                loss_x_j1 = criterion(x_j1, gt)
                loss_x_j2 = criterion(x_j2, gt)
                loss_x_j3 = criterion(x_j3, gt)
                loss_x_j4 = criterion(x_j4, gt)

                loss_vgg7 = cts(x_jf, gt, haze)
                
                loss = loss_x_jf + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 + loss_vgg7 * 0.0

            writer.add_scalar('loss/loss_x_jf', loss_x_jf, curr_iter)
            writer.add_scalar('loss/loss_x_j1', loss_x_j1, curr_iter)
            writer.add_scalar('loss/loss_x_j2', loss_x_j2, curr_iter)
            writer.add_scalar('loss/loss_x_j3', loss_x_j3, curr_iter)
            writer.add_scalar('loss/loss_x_j4', loss_x_j4, curr_iter)
            writer.add_scalar('loss/loss_vgg7', loss_vgg7, curr_iter)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            train_loss_record.update(loss.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_x_jf_record.avg,
                   loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                   optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == 1 or (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter, optimizer)
                torch.cuda.empty_cache()

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    psnr_record, ssim_record = AvgMeter(), AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data
            haze, gt = haze.cuda(), gt.cuda()

            # import pdb; pdb.set_trace()
            dehaze = sliding_forward(net, haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

            for i in range(len(haze)):
                r = dehaze[i].cpu().numpy().transpose([1, 2, 0])  # data range [0, 1]
                g = gt[i].cpu().numpy().transpose([1, 2, 0])
                psnr = peak_signal_noise_ratio(g, r)
                
                # import pdb; pdb.set_trace()
                ssim = structural_similarity(g, r, data_range=1, multichannel=True,
                                             gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
                psnr_record.update(psnr)
                ssim_record.update(ssim)
                
                
                gg = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).float()
                rr = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float()
                
                lpips = lpips_model(gg, rr).squeeze().item()
                
                writer.add_scalar('eval/psnr', psnr, curr_iter)
                writer.add_scalar('eval/ssim', ssim, curr_iter)
                writer.add_scalar('eval/lpips', lpips, curr_iter)


    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    log = '[validate]: [iter {}], [loss {:.5f}] [PSNR {:.4f}] [SSIM {:.4f}]'.format(
        curr_iter + 1, loss_record.avg, psnr_record.avg, ssim_record.avg)
    print(log)
    open(log_path, 'a').write(log + '\n')
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    # torch.cuda.set_device(int(args.gpus))

    train_dataset = OHazeDataset(OHAZE_ROOT, 'train_crop_512')
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = OHazeDataset(OHAZE_ROOT, 'val')
    val_loader = DataLoader(val_dataset, batch_size=1)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()).replace(' ', '-') + '.txt')
    
    writer = SummaryWriter(os.path.join(args.ckpt_path, args.exp_name, "logs"))
    lpips_model = LPIPS(net="alex")
    main()
