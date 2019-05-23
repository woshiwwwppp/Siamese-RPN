# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import vot
from vot import Rectangle
import sys
import os
from os.path import realpath, dirname, join
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
from time import time
from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    t1 = time()
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())
    t2 = time()
    t = t2 - t1
    print(t)



x, y, w, h = 942,603,79,79
cx=x+w//2
cy=y+h//2

path='G:\\dataset\\vot2015\\leaves'
its = os.listdir(path)
image_file=its[0]
if not image_file:
    sys.exit(0)


target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(path+'\\'+its[0])  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker




# for it in its:
#     if not it:
#          break
#     im = cv2.imread(path+'\\'+it)  # HxWxC
#     t1 = time()
#     state = SiamRPN_track(state, im)  # track
#     t2 = time()
#     t = t2 - t1
#     print(t)
#     res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
#     # print(state['score'])
#     # cv2.rectangle(im, (int(res[0]), int(res[1])), (int(res[0]+res[2]), int(res[1]+res[3])), (0, 255, 0), 2)
#     # cv2.namedWindow('res',cv2.WINDOW_NORMAL)
#     # cv2.imshow('res',im)
#     # cv2.waitKey(1)
