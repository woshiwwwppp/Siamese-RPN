# -*- coding: utf-8 -*-
import os
import random
import sys; sys.path.append('../')
import torch
# import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
import argparse
from run_SiamRPN import TrainDataLoader
# from shapely.geometry import Polygon
from tensorboardX import SummaryWriter
from os.path import realpath, dirname, join
from net import SiamRPN
from cfg import rpn

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/media/wp/software/uav_frame/00', metavar='DIR',help='path to dataset')

parser.add_argument('--weight_dir', default='/media/wp/windows/pyProject/siamese/DaSiamRPN-master_guided_anchor_train/weight', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default=None, help='resume')

parser.add_argument('--max_epoches', default=10000, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=0, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.0005, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--weight_decay', '--wd', default=5e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--debug', default=False, type=bool,  help='whether to debug')

def main():
    args = parser.parse_args()
    """ compute max_batches """
    for root, dirs, files in os.walk(args.train_path):
        for dirnames in dirs:
            dir_path = os.path.join(root, dirnames)
            args.max_batches += len(os.listdir(dir_path))

    """ Model on gpu """
    model = SiamRPN()
    model = model.cuda()
    # model.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))


    model.train().cuda()
    cudnn.benchmark = True

    """ train dataloader """
    data_loader = TrainDataLoader(args.train_path,model)
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)

    """ train phase """
    closses, rlosses, llosses = AverageMeter(), AverageMeter(), AverageMeter()
    steps = 0
    writer = SummaryWriter()
    for epoch in range(args.max_epoches):
        cur_lr = adjust_learning_rate(args.lr, optimizer, epoch, gamma=0.1)
        index_list = range(data_loader.__len__())#获取数据集的长度
        losss=[0.0,0.0,0.0]

        for example in range(args.max_batches):
            ret = data_loader.__get__(random.choice(index_list))
            template = ret['temple'].cuda()
            detection= ret['detection'].cuda()
            gtbox=list([ret['gtbox'][0].cuda()])
            shapeAdaption=list([ret['shapeAdaption'].cuda()])
            shapeAdaption1 = list([ret['shapeAdaption1'].cuda()])
            model.temple(template)
            cout,rout,lout= model(detection,shapeAdaption)
            cfg=rpn

            loss= model.loss(cout,rout,lout,shapeAdaption1,gtbox,None,ret['img_metas'],cfg,None)

            closses.update(loss['loss_cls'][0].cpu().item())
            rlosses.update(loss['loss_reg'][0].cpu().item())
            llosses.update(loss['loss_loc'][0].cpu().item())

            optimizer.zero_grad()
            loss['Loss'].backward()
            optimizer.step()
            steps += 1
            losss[0]=closses.avg
            losss[1] = rlosses.avg
            losss[2] = llosses.avg
            print("Epoch:{:04d}\tcloss:{:.4f}\trloss:{:.4f}\ttloss:{:.4f}".format(epoch,  closses.avg, rlosses.avg, llosses.avg ))
        writer.add_scalar("closses", losss[0], epoch)
        writer.add_scalar("rlosses", losss[1], epoch)
        writer.add_scalar("llosses", losss[2], epoch)
        if steps % 150 == 0:
            file_path = os.path.join(args.weight_dir, 'weights-{:07d}.pth'.format(steps))
            state = {
            'epoch' :epoch+1,
            'state_dict' :model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, file_path)

# def intersection(g, p):
#     g = Polygon(g[:8].reshape((4, 2)))
#     p = Polygon(p[:8].reshape((4, 2)))
#     if not g.is_valid or not p.is_valid:
#         return 0
#     inter = Polygon(g).intersection(Polygon(p)).area
#     union = g.area + p.area - inter
#     if union == 0:
#         return 0
#     else:
#         return inter/union
#
# def standard_nms(S, thres):
#     """ use pre_thres to filter """
#     index = np.where(S[:, 8] > thres)[0]
#     S = S[index] # ~ 100, 4
#
#     # Then use standard nms
#     order = np.argsort(S[:, 8])[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
#
#         inds = np.where(ovr <= thres)[0]
#         order = order[inds+1]
#     return S[keep]

def reshape(x):
    t = np.array(x, dtype = np.float32)
    return t.reshape(-1, 1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
