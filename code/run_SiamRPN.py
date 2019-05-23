# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import random
import os
import os.path as osp
from utils import get_subwindow_tracking
import xml.etree.ElementTree as ET
import cv2
import sys
from .net import SiamRPNvot
def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]

def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    avg_chans = np.mean(im, axis=(0, 1))#图像均值

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state

def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state

class Anchor_ms(object):
    """
    stable version for anchor generator
    """
    def __init__(self, feature_w, feature_h):
        self.w = feature_w
        self.h = feature_h
        self.base = 64  # base size for anchor box
        self.stride = 15  # center point shift stride
        self.scale = [1 / 3, 1 / 2, 1, 2, 3]  # aspect ratio
        self.anchors = self.gen_anchors()  # xywh
        self.eps = 0.01

    def gen_single_anchor(self):
        scale = np.array(self.scale, dtype=np.float32)
        s = self.base * self.base
        w, h = np.sqrt(s / scale), np.sqrt(s * scale)
        c_x, c_y = (self.stride - 1) // 2, (self.stride - 1) // 2
        anchor = np.vstack([c_x * np.ones_like(scale, dtype=np.float32), c_y * np.ones_like(scale, dtype=np.float32), w,
                            h]).transpose()
        anchor = self.center_to_corner(anchor)
        return anchor

    def gen_anchors(self):
        anchor = self.gen_single_anchor()
        k = anchor.shape[0]
        delta_x, delta_y = [x * self.stride for x in range(self.w)], [y * self.stride for y in range(self.h)]
        shift_x, shift_y = np.meshgrid(delta_x, delta_y)
        shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()
        a = shifts.shape[0]
        anchors = (anchor.reshape((1, k, 4)) + shifts.reshape((a, 1, 4))).reshape((a * k, 4))  # corner format
        anchors = self.corner_to_center(anchors)
        return anchors

    # float
    def diff_anchor_gt(self, gt):
        eps = self.eps
        anchors, gt = self.anchors.copy(), gt.copy()
        diff = np.zeros_like(anchors, dtype=np.float32)  # 生成和anchor一样大小的0矩阵
        # detection帧的groundtruth，四个坐标值分别反变换后得到rpn滑框的dx,dy,dw,dh
        diff[:, 0] = (gt[0] - anchors[:, 0]) / (anchors[:, 2] + eps)  #
        diff[:, 1] = (gt[1] - anchors[:, 1]) / (anchors[:, 3] + eps)
        diff[:, 2] = np.log((gt[2] + eps) / (anchors[:, 2] + eps))
        diff[:, 3] = np.log((gt[3] + eps) / (anchors[:, 3] + eps))
        return diff

    # float
    def center_to_corner(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype=np.float32)
        box_[:, 0] = box[:, 0] - (box[:, 2] - 1) / 2
        box_[:, 1] = box[:, 1] - (box[:, 3] - 1) / 2
        box_[:, 2] = box[:, 0] + (box[:, 2] - 1) / 2
        box_[:, 3] = box[:, 1] + (box[:, 3] - 1) / 2
        box_ = box_.astype(np.float32)
        return box_

    # float
    def corner_to_center(self, box):
        box = box.copy()
        box_ = np.zeros_like(box, dtype=np.float32)
        box_[:, 0] = box[:, 0] + (box[:, 2] - box[:, 0]) / 2
        box_[:, 1] = box[:, 1] + (box[:, 3] - box[:, 1]) / 2
        box_[:, 2] = (box[:, 2] - box[:, 0])
        box_[:, 3] = (box[:, 3] - box[:, 1])
        box_ = box_.astype(np.float32)
        return box_

    def pos_neg_anchor(self, gt, pos_num=16, neg_num=48, threshold_pos=0.5, threshold_neg=0.1):
        #根据detection帧的box与自动生成的box计算iou，得出正负样本
        gt = gt.copy()
        gt_corner = self.center_to_corner(np.array(gt, dtype=np.float32).reshape(1, 4))
        an_corner = self.center_to_corner(np.array(self.anchors, dtype=np.float32))
        # 计算groundtruth与生成的1445个框的iou
        iou_value = self.iou(an_corner, gt_corner).reshape(-1)  # (1445)
        max_iou = max(iou_value)
        pos, neg = np.zeros_like(iou_value, dtype=np.int32), np.zeros_like(iou_value, dtype=np.int32)

        # pos
        # 选取得分最高的30个框
        pos_cand = np.argsort(iou_value)[::-1][:30]
        # 从这30个框中选择16个作为正样本
        pos_index = np.random.choice(pos_cand, pos_num, replace=False)
        if max_iou > threshold_pos:
            pos[pos_index] = 1  # 对应pos_index的pos置1

        # neg
        neg_cand = np.where(iou_value < threshold_neg)[0]  # 选择iou小于threshold_neg的为负样本
        neg_ind = np.random.choice(neg_cand, neg_num, replace=False)  # 随机选择48个负样本
        neg[neg_ind] = 1

        return pos, neg

    def iou(self, box1, box2):
        box1, box2 = box1.copy(), box2.copy()
        N = box1.shape[0]
        K = box2.shape[0]
        box1 = np.array(box1.reshape((N, 1, 4))) + np.zeros((1, K, 4))  # box1=[N,K,4]
        box2 = np.array(box2.reshape((1, K, 4))) + np.zeros((N, 1, 4))  # box1=[N,K,4]
        x_max = np.max(np.stack((box1[:, :, 0], box2[:, :, 0]), axis=-1), axis=2)
        x_min = np.min(np.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
        y_max = np.max(np.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
        y_min = np.min(np.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)
        tb = x_min - x_max
        lr = y_min - y_max
        tb[np.where(tb < 0)] = 0
        lr[np.where(lr < 0)] = 0
        over_square = tb * lr
        all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (
                    box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1]) - over_square
        return over_square / all_square

class TrainDataLoader(object):

    # out_feature最后滑框和类别特征图的大小
    def __init__(self, img_dir_path, net,out_feature=19, max_inter=80, check=False ,tmp_dir='../tmp/visualization'):  #
        assert osp.isdir(img_dir_path), 'input img_dir_path error'

        self.img_dir_path = img_dir_path  # this is a root dir contain subclass
        self.max_inter = max_inter
        self.sub_class_dir = [sub_class_dir for sub_class_dir in os.listdir(img_dir_path) if
                              os.path.isdir(os.path.join(img_dir_path, sub_class_dir))]
        self.anchor_generator = Anchor_ms(out_feature, out_feature)
        self.anchors = self.anchor_generator.gen_anchors()  # centor，依据17*17的特征图，每个点生成5个框，17*17*5=1445
        self.ret = {}
        self.check = check
        self.tmp_dir = self.init_dir(tmp_dir)
        self.ret['tmp_dir'] = tmp_dir
        self.ret['check'] = check
        self.count = 0
        self.ret['p']=TrackerConfig()
        self.ret['p'].update(net.cfg)

    def init_dir(self, tmp_dir):
        if not osp.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    # def get_transform_for_train(self):
    #     transform_list = []
    #     transform_list.append(transforms.ToTensor())
    #     transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    #     return transforms.Compose(transform_list)
    #
    # # tuple  计算图像三个通道的均值
    # def _average(self):
    #     assert self.ret.__contains__('template_img_path'), 'no template path'
    #     assert self.ret.__contains__('detection_img_path'), 'no detection path'
    #     template = Image.open(self.ret['template_img_path'])
    #     detection = Image.open(self.ret['detection_img_path'])
    #
    #     mean_template = tuple(map(round, ImageStat.Stat(template).mean))
    #     mean_detection = tuple(map(round, ImageStat.Stat(detection).mean))
    #     self.ret['mean_template'] = mean_template
    #     self.ret['mean_detection'] = mean_detection

    def _pick_img_pairs(self, index_of_subclass):
        # img_dir_path -> sub_class_dir_path -> template_img_path
        # use index_of_subclass to select a sub directory
        assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'
        sub_class_dir_basename = self.sub_class_dir[index_of_subclass]
        sub_class_dir_path = os.path.join(self.img_dir_path, sub_class_dir_basename)
        sub_class_img_name = [img_name for img_name in os.listdir(sub_class_dir_path) if
                              not img_name.find('.jpg') == -1]
        sub_class_img_name = sorted(sub_class_img_name)
        sub_class_img_num = len(sub_class_img_name)
        # select template, detection
        # ++++++++++++++++++++++++++++ add break in sequeence [0,0,0,0] ++++++++++++++++++++++++++++++++++
        if self.max_inter >= sub_class_img_num - 1:
            self.max_inter = sub_class_img_num // 2
        template_index = np.clip(random.choice(range(0, max(1, sub_class_img_num - self.max_inter))), 0,
                                 sub_class_img_num - 1)  # 在子文件夹中随机选了一个template
        detection_index = np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0,
                                  sub_class_img_num - 1)  # 在子文件夹中随机选了一个template
        template_name, detection_name = sub_class_img_name[template_index], sub_class_img_name[detection_index]
        template_img_path, detection_img_path = osp.join(sub_class_dir_path, template_name), osp.join(
            sub_class_dir_path, detection_name)
        im_template = cv2.imread(template_img_path)
        im_detection = cv2.imread(detection_img_path)
        template_xml_path=template_img_path.split('.')[0]+'.xml'
        detection_xml_path = detection_img_path.split('.')[0] + '.xml'
        box_template=self.get_xywh_from_xml(template_xml_path)
        box_detection = self.get_xywh_from_xml(detection_xml_path)

        # load infomation of template and detection
        self.ret['img_template'] = im_template
        self.ret['img_detection'] = im_detection
        self.ret['template_target_pos'] = np.array([box_template[0],box_template[1]])
        self.ret['template_target_sz'] = np.array([box_template[2],box_template[3]])
        self.ret['detection_target_pos'] = np.array([box_detection[0],box_detection[1]])
        self.ret['detection_target_sz'] = np.array([box_detection[2], box_detection[3]])
        self.ret['anchors'] = self.anchors
        # self._average()  # 计算图像均值

    def get_xywh_from_xml(self,file):
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        x,y,w,h=0,0,0,0
        for object in root.iter('object'):
            bndbox=object.find('bndbox')
            xmin= int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            x=int((xmin+xmax)/2)
            y=int((ymin+ymax)/2)
            w=int(xmax-xmin)
            h=int(ymax-ymin)
        return [x,y,w,h]

    def _pad_crop_resize_template(self):
        self.ret['im_h'] = self.ret['img_template'].shape[0]
        self.ret['im_w'] = self.ret['img_template'].shape[1]
        self.ret['p'].score_size = (self.ret['p'].instance_size - self.ret['p'].exemplar_size) / self.ret['p'].total_stride + 1
        self.ret['p'].anchor = generate_anchor(self.ret['p'].total_stride, self.ret['p'].scales, self.ret['p'].ratios, int(self.ret['p'].score_size))
        avg_chans = np.mean(self.ret['img_template'], axis=(0, 1))  # 图像均值
        wc_z = self.ret['template_target_sz'][0] + self.ret['p'].context_amount * sum(self.ret['template_target_sz'])
        hc_z = self.ret['template_target_sz'][1] + self.ret['p'].context_amount * sum(self.ret['template_target_sz'])
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(self.ret['img_template'], self.ret['template_target_pos'], self.ret['p'].exemplar_size, s_z, avg_chans)
        z = Variable(z_crop.unsqueeze(0))
        # net.temple(z.cuda())
        if self.ret['p'].windowing == 'cosine':
            window = np.outer(np.hanning(self.ret['p'].score_size), np.hanning(self.ret['p'].score_size))
        elif self.ret['p'].windowing == 'uniform':
            window = np.ones((self.ret['p'].score_size, self.ret['p'].score_size))
        window = np.tile(window.flatten(), self.ret['p'].anchor_num)
        self.ret['temple'] = z
        self.ret['avg_chans'] = avg_chans
        self.ret['window'] = window

    def _pad_crop_resize_detection(self):
        wc_z = self.ret['detection_target_sz'][1] + self.ret['p'].context_amount * sum(self.ret['detection_target_sz'])
        hc_z = self.ret['detection_target_sz'][0] + self.ret['p'].context_amount * sum(self.ret['detection_target_sz'])
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.ret['p'].exemplar_size / s_z
        d_search = (self.ret['p'].instance_size - self.ret['p'].exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        avg_chans = np.mean(self.ret['img_detection'], axis=(0, 1))  # 图像均值
        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(self.ret['img_detection'], self.ret['detection_target_pos'], self.ret['p'].instance_size, round(s_x), avg_chans).unsqueeze(0))
        self.ret['detection'] = x_crop

    def _generate_pos_neg_diff(self):

        # np.array((self.ret['detection_target_pos'][0],self.ret['detection_target_pos'][1], self.ret['detection_target_sz'][0], self.ret['detection_target_sz'][1]), dtype=np.int32)
        gt_box_in_detection = np.array((self.ret['detection_target_pos'][0],self.ret['detection_target_pos'][1], self.ret['detection_target_sz'][0], self.ret['detection_target_sz'][1]), dtype=np.int32)
        pos, neg = self.anchor_generator.pos_neg_anchor(gt_box_in_detection)  # 生成框的正负样本标记向量
        diff = self.anchor_generator.diff_anchor_gt(gt_box_in_detection)  # 生成回归框的dx,dy,dw,dh
        pos, neg, diff = pos.reshape((-1, 1)), neg.reshape((-1, 1)), diff.reshape((-1, 4))
        class_target = np.array([-100.] * self.anchors.shape[0], np.int32)

        # pos   获取pos的真实长度，并生成标记向量
        pos_index = np.where(pos == 1)[0]
        pos_num = len(pos_index)
        self.ret['pos_anchors'] = np.array(self.ret['anchors'][pos_index, :],dtype=np.int32) if not pos_num == 0 else None
        if pos_num > 0:
            class_target[pos_index] = 1

        # neg   获取neg的真实长度，并生成标记向量
        neg_index = np.where(neg == 1)[0]
        neg_num = len(neg_index)
        class_target[neg_index] = 0

        class_logits = class_target.reshape(-1, 1)  # 生成框的类别logits的值用于分类，负样本的值为-100，正样本的值为0
        pos_neg_diff = np.hstack((class_logits, diff))  # 将类别和坐标合并
        self.ret['pos_neg_diff'] = pos_neg_diff
        self.ret['pos_neg_diff_tensor'] = torch.Tensor(pos_neg_diff)

        return pos_neg_diff

    def to_torch(self,ndarray):
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor"
                             .format(type(ndarray)))
        return ndarray

    def im_to_torch(self,img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = self.to_torch(img).float()
        return img

    def get_subwindow_tracking(self,im, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        # zzp: a more easy speed version
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                             np.uint8)  # 0 is better than 1 initialization
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1),
                                :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original

        return self.im_to_torch(im_patch) if out_mode in 'torch' else im_patch

    def __get__(self, index):
        self._pick_img_pairs(index)
        self._pad_crop_resize_template()
        self._pad_crop_resize_detection()
        self._generate_pos_neg_diff()  # 生成框的1445*5的张量，代表每个框的类别，dx,dy,dw,dh
        # self._tranform()  # PIL to Tensor
        self.count += 1
        return self.ret

    def __len__(self):
        return len(self.sub_class_dir)



if __name__ == '__main__':
    # we will do a test for dataloader
    net = SiamRPNvot()
    loader = TrainDataLoader('D:\\uav_frame\\00',net ,check = True)
    #print(loader.__len__())
    index_list = range(loader.__len__())
    for i in range(1000):
        ret = loader.__get__(random.choice(index_list))
        label = ret['pos_neg_diff'][:, 0].reshape(-1)
        pos_index = list(np.where(label == 1)[0])
        pos_num = len(pos_index)
        print(pos_index)
        print(pos_num)
        if pos_num != 0 and pos_num != 16:
            print(pos_num)
            sys.exit(0)
        print(i)
