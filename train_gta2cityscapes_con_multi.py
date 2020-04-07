import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

import time
from PIL import Image

import torchvision
from torchvision import transforms, utils
from tensorboardX import SummaryWriter


from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
# from dataset.gta5_dataset import GTA5DataSet
# from dataset.cityscapes_dataset import cityscapesDataSet
from augmentation import RandomCrop, RandAugment, Cutout
from datasets import GTA5, Cityscapes, print_palette
from utils_ import *

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '../DomainAdaptationConsistency/data/gtav2cityscapes/gtav/'
DATA_LIST_PATH = '../DomainAdaptationConsistency/data/gtav2cityscapes/gtav/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = '../DomainAdaptationConsistency/data/gtav2cityscapes/cityscapes/'
DATA_LIST_PATH_TARGET = '../DomainAdaptationConsistency/data/gtav2cityscapes/cityscapes/cityscapes_list/'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = './pretrain/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM = './pretrain/resnet101-caffe.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/con_multi/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
LAMBDA_CON = 0.1
TAU = 0.95
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-con", type=float, default=LAMBDA_CON,
                        help="lambda_con for consistent training.")
    parser.add_argument("--tau", type=float, default=TAU,
                        help="tau for psuedo-label threshold.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(reduction='mean').cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle']
num_cls = len(classes)

class Interpolate(nn.Module):
    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode
        self.align_corners =align_corners
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x
    
# path_detail = 'batch_{}_thre_{}_lamb_{}'.format(args.batch_size,
#                                                 '{:0.2f}'.format(args.tau).replace('.', ''),
#                                                 '{:0.3f}'.format(args.lambda_con).replace('.', ''))
path_detail = 'cons_{}_adv1_{}_adv2_{}'.format('{:0.4f}'.format(args.lambda_con).replace('.', ''),
                                               '{:0.4f}'.format(args.lambda_adv_target1).replace('.', ''),
                                               '{:0.4f}'.format(args.lambda_adv_target2).replace('.', ''))
writer = {
          'train': SummaryWriter(log_dir='runs/train_Con_multi_'+path_detail),
          'test':  SummaryWriter(log_dir='runs/test_Con_multi_'+path_detail)
         }

def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu
    
    tau = torch.ones(1) * args.tau
    tau = tau.cuda(args.gpu)

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params, False)
    elif args.model == 'DeepLabVGG':
        model = DeeplabVGG(pretrained=True, num_classes=args.num_classes)
    
    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    weak_transform = transforms.Compose(
        [
    #         transforms.RandomCrop(32, 4),
    #         transforms.RandomRotation(30),
    #         transforms.Resize(1024),
            transforms.ToTensor(), 
    #         transforms.Normalize(mean, std),
    #         RandomCrop(768)
        ])

    target_transform = transforms.Compose(
        [
    #         transforms.RandomCrop(32, 4),
    #         transforms.RandomRotation(30),
    #         transforms.Normalize(mean, std)
    #         transforms.Resize(1024),
    #         transforms.ToTensor(), 
    #         RandomCrop(768)
        ])
    
    label_set = GTA5(root=args.data_dir, num_cls=19, split='all', remap_labels=True,
                     transform=weak_transform, 
                     target_transform=target_transform,
                     scale=input_size,
        #              crop_transform=RandomCrop(int(768*(args.scale/1024))),
                     )
    unlabel_set = Cityscapes(root=args.data_dir_target, split=args.set, remap_labels=True, 
                             transform=weak_transform,
                             target_transform=target_transform,
                             scale=input_size_target,
#                              crop_transform=RandomCrop(int(768*(args.scale/1024))),
                            )

    test_set = Cityscapes(root=args.data_dir_target, split='val', remap_labels=True, 
                          transform=weak_transform,
                          target_transform=target_transform,
                          scale=input_size_target,
    #                       crop_transform=RandomCrop(768)
                         )

    
    label_loader = data.DataLoader(label_set, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=args.num_workers, pin_memory=False)
    
    unlabel_loader = data.DataLoader(unlabel_set, batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers, pin_memory=False)
    
    test_loader = data.DataLoader(test_set, batch_size=2, shuffle=False, 
                                  num_workers=args.num_workers, pin_memory=False)


    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    interp = Interpolate(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = Interpolate(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    interp_test = Interpolate(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
#     interp_test = Interpolate(size=(1024, 2048), mode='bilinear', align_corners=True)
    
    normalize_transform = transforms.Compose([
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])
    
    # labels for adversarial training
    source_label = 0
    target_label = 1
    
    max_mIoU = 0
    
    total_loss_seg_value1 = []
    total_loss_adv_target_value1 = []
    total_loss_D_value1 = []
    total_loss_con_value1 = []
    
    total_loss_seg_value2 = []
    total_loss_adv_target_value2 = []
    total_loss_D_value2 = []
    total_loss_con_value2 = []
    
    hist = np.zeros((num_cls, num_cls))

#     for i_iter in range(args.num_steps):
    for i_iter, (batch, batch_un) in enumerate(zip(roundrobin_infinite(label_loader), roundrobin_infinite(unlabel_loader))):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0
        loss_con_value1 = 0
        
        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0
        loss_con_value2 = 0
        
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        # train G

        # don't accumulate grads in D
        for param in model_D1.parameters():
            param.requires_grad = False
            
        for param in model_D2.parameters():
                param.requires_grad = False

        # train with source

        images, labels = batch
        images_orig = images
        images = transform_batch(images, normalize_transform)
        images = Variable(images).cuda(args.gpu)

        pred1, pred2 = model(images)
        pred1 = interp(pred1)
        pred2 = interp(pred2)
        
        loss_seg1 = loss_calc(pred1, labels, args.gpu)
        loss_seg2 = loss_calc(pred2, labels, args.gpu)
        loss = loss_seg2 + args.lambda_seg * loss_seg1

        # proper normalization
        loss = loss / args.iter_size
        loss.backward()
        loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
        loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

        # train with target

        images_tar, labels_tar = batch_un
        images_tar_orig = images_tar
        images_tar = transform_batch(images_tar, normalize_transform)
        images_tar = Variable(images_tar).cuda(args.gpu)

        pred_target1, pred_target2 = model(images_tar)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        D_out1 = model_D1(F.softmax(pred_target1, dim=1))
        D_out2 = model_D2(F.softmax(pred_target2, dim=1))
        
        loss_adv_target1 = bce_loss(D_out1,
                                    Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                        args.gpu))
        
        loss_adv_target2 = bce_loss(D_out2,
                                    Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                        args.gpu))

        loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
        loss = loss / args.iter_size
        loss.backward()
        loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
        loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / args.iter_size

        # train with consistency loss
        # unsupervise phase
        policies = RandAugment().get_batch_policy(args.batch_size)
        rand_p1 = np.random.random(size=args.batch_size)
        rand_p2 = np.random.random(size=args.batch_size)
        random_dir = np.random.choice([-1, 1], size=[args.batch_size, 2])
        
        images_aug = aug_batch_tensor(images_tar_orig, policies, rand_p1, rand_p2, random_dir)
        
        images_aug_orig = images_aug
        images_aug = transform_batch(images_aug, normalize_transform)
        images_aug = Variable(images_aug).cuda(args.gpu)
         
        pred_target_aug1, pred_target_aug2 = model(images_aug)
        pred_target_aug1 = interp_target(pred_target_aug1)
        pred_target_aug2 = interp_target(pred_target_aug2)
        
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        
        max_pred1, psuedo_label1 = torch.max(F.softmax(pred_target1, dim=1), 1)
        max_pred2, psuedo_label2 = torch.max(F.softmax(pred_target2, dim=1), 1)
        
        psuedo_label1 = psuedo_label1.cpu().numpy().astype(np.float32)
        psuedo_label1_thre = psuedo_label1.copy()
        psuedo_label1_thre[(max_pred1 < tau).cpu().numpy().astype(np.bool)] = 255 # threshold to don't care
        psuedo_label1_thre = aug_batch_numpy(psuedo_label1_thre, policies, rand_p1, rand_p2, random_dir)
        psuedo_label2 = psuedo_label2.cpu().numpy().astype(np.float32)
        psuedo_label2_thre = psuedo_label2.copy()
        psuedo_label2_thre[(max_pred2 < tau).cpu().numpy().astype(np.bool)] = 255 # threshold to don't care
        psuedo_label2_thre = aug_batch_numpy(psuedo_label2_thre, policies, rand_p1, rand_p2, random_dir)

        psuedo_label1_thre = Variable(psuedo_label1_thre).cuda(args.gpu)
        psuedo_label2_thre = Variable(psuedo_label2_thre).cuda(args.gpu)
        
        if (psuedo_label1_thre != 255).sum().cpu().numpy() > 0:
            # nll_loss doesn't support empty tensors 
            loss_con1 = loss_calc(pred_target_aug1, psuedo_label1_thre, args.gpu)
            loss_con_value1 += loss_con1.data.cpu().numpy() / args.iter_size
        else:
            loss_con1 = torch.tensor(0.0, requires_grad=True).cuda(args.gpu)
            
        if (psuedo_label2_thre != 255).sum().cpu().numpy() > 0:
            # nll_loss doesn't support empty tensors 
            loss_con2 = loss_calc(pred_target_aug2, psuedo_label2_thre, args.gpu)
            loss_con_value2 += loss_con2.data.cpu().numpy() / args.iter_size
        else:
            loss_con2 = torch.tensor(0.0, requires_grad=True).cuda(args.gpu)
            
        loss = args.lambda_con * loss_con1 + args.lambda_con * loss_con2
        # proper normalization
        loss = loss / args.iter_size
        loss.backward()
        
        # train D

        # bring back requires_grad
        for param in model_D1.parameters():
            param.requires_grad = True
        
        for param in model_D2.parameters():
            param.requires_grad = True

        # train with source
        pred1 = pred1.detach()
        pred2 = pred2.detach()

        D_out1 = model_D1(F.softmax(pred1, dim=1))
        D_out2 = model_D2(F.softmax(pred2, dim=1))

        loss_D1 = bce_loss(D_out1,
                          Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_D2 = bce_loss(D_out2,
                           Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_D1 = loss_D1 / args.iter_size / 2
        loss_D2 = loss_D2 / args.iter_size / 2

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.data.cpu().numpy()
        loss_D_value2 += loss_D2.data.cpu().numpy()

        # train with target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        D_out1 = model_D1(F.softmax(pred_target1, dim=1))
        D_out2 = model_D2(F.softmax(pred_target2, dim=1))

        loss_D1 = bce_loss(D_out1,
                           Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

        loss_D2 = bce_loss(D_out2,
                           Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(args.gpu))

        loss_D1 = loss_D1 / args.iter_size / 2
        loss_D2 = loss_D2 / args.iter_size / 2

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.data.cpu().numpy()
        loss_D_value2 += loss_D2.data.cpu().numpy()
            
        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        
        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}, loss_con1 = {8:.3f}, loss_con2 = {9:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, loss_con_value1, loss_con_value2))
        
        total_loss_seg_value1.append(loss_seg_value1)
        total_loss_adv_target_value1.append(loss_adv_target_value1)
        total_loss_D_value1.append(loss_D_value1)
        total_loss_con_value1.append(loss_con_value1)
        
        total_loss_seg_value2.append(loss_seg_value2)
        total_loss_adv_target_value2.append(loss_adv_target_value2)
        total_loss_D_value2.append(loss_D_value2)
        total_loss_con_value2.append(loss_con_value2)
        
        hist += fast_hist(labels.cpu().numpy().flatten().astype(int), 
                          torch.argmax(pred2, dim=1).cpu().numpy().flatten().astype(int), 
                          num_cls)

        if i_iter % 10 == 0:
            print('({}/{})'.format(i_iter+1, int(args.num_steps)))
            acc_overall, acc_percls, iu, fwIU = result_stats(hist)
            mIoU = np.mean(iu)
            per_class = [[classes[i], acc] for i, acc in list(enumerate(iu))]
            per_class = np.array(per_class).flatten()
            print(('per cls IoU :'+('\n{:>14s} : {}')*19) .format(*per_class))
            print('mIoU : {:0.2f}'.format(np.mean(iu)))
            print('fwIoU : {:0.2f}'.format(fwIU))
            print('pixel acc : {:0.2f}'.format(acc_overall))
            per_class = [[classes[i], acc] for i, acc in list(enumerate(acc_percls))]
            per_class = np.array(per_class).flatten()
            print(('per cls acc :'+('\n{:>14s} : {}')*19) .format(*per_class))

            avg_train_acc       = acc_overall
            avg_train_loss_seg1 = np.mean(total_loss_seg_value1)
            avg_train_loss_adv1 = np.mean(total_loss_adv_target_value1)
            avg_train_loss_dis1 = np.mean(total_loss_D_value1)
            avg_train_loss_con1 = np.mean(total_loss_con_value1)
            avg_train_loss_seg2 = np.mean(total_loss_seg_value2)
            avg_train_loss_adv2 = np.mean(total_loss_adv_target_value2)
            avg_train_loss_dis2 = np.mean(total_loss_D_value2)
            avg_train_loss_con2 = np.mean(total_loss_con_value2)

            print('avg_train_acc      :', avg_train_acc      )
            print('avg_train_loss_seg1 :', avg_train_loss_seg1 )
            print('avg_train_loss_adv1 :', avg_train_loss_adv1 )
            print('avg_train_loss_dis1 :', avg_train_loss_dis1 )
            print('avg_train_loss_con1 :', avg_train_loss_con1 )
            print('avg_train_loss_seg2 :', avg_train_loss_seg2 )
            print('avg_train_loss_adv2 :', avg_train_loss_adv2 )
            print('avg_train_loss_dis2 :', avg_train_loss_dis2 )
            print('avg_train_loss_con2 :', avg_train_loss_con2 )
            
            writer['train'].add_scalar('log/mIoU',      mIoU                , i_iter)
            writer['train'].add_scalar('log/acc',       avg_train_acc       , i_iter)
            writer['train'].add_scalar('log1/loss_seg', avg_train_loss_seg1 , i_iter)
            writer['train'].add_scalar('log1/loss_adv', avg_train_loss_adv1 , i_iter)
            writer['train'].add_scalar('log1/loss_dis', avg_train_loss_dis1 , i_iter)
            writer['train'].add_scalar('log1/loss_con', avg_train_loss_con1 , i_iter)
            writer['train'].add_scalar('log2/loss_seg', avg_train_loss_seg2 , i_iter)
            writer['train'].add_scalar('log2/loss_adv', avg_train_loss_adv2 , i_iter)
            writer['train'].add_scalar('log2/loss_dis', avg_train_loss_dis2 , i_iter)
            writer['train'].add_scalar('log2/loss_con', avg_train_loss_con2 , i_iter)
            
            hist = np.zeros((num_cls, num_cls))
            total_loss_seg_value1 = []
            total_loss_adv_target_value1 = []
            total_loss_D_value1 = []
            total_loss_con_value1 = []
            
            fig = plt.figure(figsize=(15, 15))

            labels = labels[0].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(331)
            ax.imshow(print_palette(Image.fromarray(labels).convert('L')))
            ax.axis("off")
            ax.set_title('labels')
            
            ax = fig.add_subplot(337)
            images = images_orig[0].cpu().numpy().transpose((1, 2, 0))
#             images += IMG_MEAN
            ax.imshow(images)
            ax.axis("off")
            ax.set_title('datas')
    
            _, pred2 = torch.max(pred2, dim=1)
            pred2 = pred2[0].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(334)
            ax.imshow(print_palette(Image.fromarray(pred2).convert('L')))
            ax.axis("off")
            ax.set_title('predicts')
            
            labels_tar = labels_tar[0].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(332)
            ax.imshow(print_palette(Image.fromarray(labels_tar).convert('L')))
            ax.axis("off")
            ax.set_title('tar_labels')

            ax = fig.add_subplot(338)
            ax.imshow(images_tar_orig[0].cpu().numpy().transpose((1, 2, 0)))
            ax.axis("off")
            ax.set_title('tar_datas')
            
            _, pred_target2 = torch.max(pred_target2, dim=1)
            pred_target2 = pred_target2[0].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(335)
            ax.imshow(print_palette(Image.fromarray(pred_target2).convert('L')))
            ax.axis("off")
            ax.set_title('tar_predicts')
            
            print(policies[0], 'p1', rand_p1[0], 'p2', rand_p2[0], 'random_dir', random_dir[0])
            
            psuedo_label2_thre = psuedo_label2_thre[0].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(333)
            ax.imshow(print_palette(Image.fromarray(psuedo_label2_thre).convert('L')))
            ax.axis("off")
            ax.set_title('psuedo_labels')

            ax = fig.add_subplot(339)
            ax.imshow(images_aug_orig[0].cpu().numpy().transpose((1, 2, 0)))
            ax.axis("off")
            ax.set_title('aug_datas')
            
            _, pred_target_aug2 = torch.max(pred_target_aug2, dim=1)
            pred_target_aug2 = pred_target_aug2[0].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(336)
            ax.imshow(print_palette(Image.fromarray(pred_target_aug2).convert('L')))
            ax.axis("off")
            ax.set_title('aug_predicts')

#             plt.show()
            writer['train'].add_figure('image/', 
                                       fig, 
                                       global_step=i_iter,
                                       close=True
                                      )

        if i_iter % 500 == 0:
            loss1 = []
            loss2 = []
            for test_i, batch in enumerate(test_loader):

                images, labels = batch
                images_orig = images
                images = transform_batch(images, normalize_transform)
                images = Variable(images).cuda(args.gpu)

                pred1, pred2 = model(images)
                pred1 = interp_test(pred1)
                pred1 = pred1.detach()
                pred2 = interp_test(pred2)
                pred2 = pred2.detach()

                loss_seg1 = loss_calc(pred1, labels, args.gpu)
                loss_seg2 = loss_calc(pred2, labels, args.gpu)
                loss1.append(loss_seg1.item())
                loss2.append(loss_seg2.item())
                
                hist += fast_hist(labels.cpu().numpy().flatten().astype(int), 
                                  torch.argmax(pred2, dim=1).cpu().numpy().flatten().astype(int), 
                                  num_cls)  
                
            print('test')
            fig = plt.figure(figsize=(15, 15))
            labels = labels[-1].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(311)
            ax.imshow(print_palette(Image.fromarray(labels).convert('L')))
            ax.axis("off")
            ax.set_title('labels')

            ax = fig.add_subplot(313)
            ax.imshow(images_orig[-1].cpu().numpy().transpose((1, 2, 0)))
            ax.axis("off")
            ax.set_title('datas')

            _, pred2 = torch.max(pred2, dim=1)
            pred2 = pred2[-1].cpu().numpy().astype(np.float32)
            ax = fig.add_subplot(312)
            ax.imshow(print_palette(Image.fromarray(pred2).convert('L')))
            ax.axis("off")
            ax.set_title('predicts')

#             plt.show()

            writer['test'].add_figure('test_image/', 
                                      fig, 
                                      global_step=i_iter,
                                      close=True
                                     )
            
            acc_overall, acc_percls, iu, fwIU = result_stats(hist)
            mIoU = np.mean(iu)
            per_class = [[classes[i], acc] for i, acc in list(enumerate(iu))]
            per_class = np.array(per_class).flatten()
            print(('per cls IoU :'+('\n{:>14s} : {}')*19) .format(*per_class))
            print('mIoU : {:0.2f}'.format(mIoU))
            print('fwIoU : {:0.2f}'.format(fwIU))
            print('pixel acc : {:0.2f}'.format(acc_overall))
            per_class = [[classes[i], acc] for i, acc in list(enumerate(acc_percls))]
            per_class = np.array(per_class).flatten()
            print(('per cls acc :'+('\n{:>14s} : {}')*19) .format(*per_class))

            avg_test_loss1 = np.mean(loss1)
            avg_test_loss2 = np.mean(loss2)
            avg_test_acc  = acc_overall
            print('avg_test_loss2 :', avg_test_loss1)
            print('avg_test_loss1 :', avg_test_loss2)
            print('avg_test_acc   :', avg_test_acc  )
            writer['test'].add_scalar('log1/loss_seg', avg_test_loss1, i_iter)
            writer['test'].add_scalar('log2/loss_seg', avg_test_loss2, i_iter)
            writer['test'].add_scalar('log/acc',       avg_test_acc  , i_iter)
            writer['test'].add_scalar('log/mIoU',      mIoU          , i_iter)
            
            hist = np.zeros((num_cls, num_cls))

            
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir,    'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if max_mIoU < mIoU:
            max_mIoU = mIoU
            torch.save(model.state_dict(), osp.join(args.snapshot_dir,    'GTA5_' + 'best_iter' + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + 'best_iter' + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + 'best_iter' + '_D2.pth'))
            
#         if i_iter % args.save_pred_every == 0 and i_iter != 0:
#             print('taking snapshot ...')
#             torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
#             torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
#             torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))
        
if __name__ == '__main__':
    main()
    
# python train_gta2cityscapes_con_multi.py --snapshot-dir=./snapshots/con_multi_con0001/ --lambda-con=0.001 --gpu=1 --lambda-adv-target2=
