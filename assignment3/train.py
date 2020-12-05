import sys
root = '/st2/hyewon/Trials/CS576/assignment3/'
sys.path.append(root)

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from yolo import Yolo
from data import VOCDetection
from loss import Loss
from utils import inference

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# Configurations
run_name = 'vgg16'          # experiment name.
ckpt_root = 'checkpoints_200_new'   # from/to which directory to load/save checkpoints.
data_root = '/st2/hyewon/Trials/CS576/assignment3/dataset'       # where the data exists.
pretrained_backbone_path = '/st2/hyewon/Trials/CS576/assignment3/weights/vgg_features.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001          # learning rate
batch_size = 64     # batch_size
last_epoch = 1      # the last training epoch. (defulat: 1)
max_epoch = 115     # maximum epoch for the training.

num_boxes = 2       # the number of boxes for each grid in Yolo v1.
num_classes = 20    # the number of classes in Pascal VOC Detection.
grid_size = 7       # 3x224x224 image is reduced to (5*num_boxes+num_classes)x7x7.
lambda_coord = 7    # weight for coordinate regression loss.
lambda_noobj = 0.5  # weight for no-objectness confidence loss.

ckpt_dir = os.path.join(root, ckpt_root)
makedirs(ckpt_dir)
# !ln -s '/gdrive/My Drive'/{ckpt_dir.replace('/gdrive/My Drive/', '')} ./

train_dset = VOCDetection(root=data_root, split='train')
train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

test_dset = VOCDetection(root=data_root, split='test')
test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

model = Yolo(grid_size, num_boxes, num_classes)
model = model.to(device)
pretrained_weights = torch.load(pretrained_backbone_path)
model.load_state_dict(pretrained_weights)
print('loaded pretrained weight')

# Freeze the backbone network.
model.features.requires_grad_(False)
model_params = [v for v in model.parameters() if v.requires_grad is True]
optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=5e-4)
compute_loss = Loss(grid_size, num_boxes, num_classes)

# Load the last checkpoint if exits.
ckpt_path = os.path.join(ckpt_dir, 'last.pth') 

if os.path.exists(ckpt_path): 
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    last_epoch = ckpt['epoch'] + 1
    print('Last checkpoint is loaded. start_epoch:', last_epoch)
else:
    print('No checkpoint is found.')

# Training & Testing.
model = model.to(device)
num_iter = 0
for epoch in range(1, max_epoch):
    total_loss = 0
    # Learning rate scheduling
    if epoch in [50, 150]:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch < last_epoch:
        continue

    model.train()
    for i, (x, y) in enumerate(train_dloader):
        # implement training pipeline here
        x = x.to(device)
        y = y.to(device)
        
        logit = model(x)
        # loss = compute_loss(logit, y)
        loss_xy, loss_wh, loss_obj, loss_noobj, loss_class = compute_loss(logit, y)
        # print (loss_xy, loss_wh, loss_obj, loss_noobj, loss_class)
        loss = (lambda_coord*(loss_xy + loss_wh) + loss_obj + lambda_noobj * loss_noobj + loss_class)/batch_size
        total_loss += loss
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, max_epoch, i+1, len(train_dloader), loss, total_loss / (i+1)))
            num_iter += 1
            
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        test_accuracy = 0
        test_num_data = 0
        for x, y in test_dloader:
            # implement testing pipeline here
            x = x.to(device)
            y = y.to(device)
            
            logit = model(x)
            loss_xy, loss_wh, loss_obj, loss_noobj, loss_class = compute_loss(logit, y)
            loss = (lambda_coord*(loss_xy + loss_wh) + loss_obj + lambda_noobj * loss_noobj + loss_class)/batch_size
            valid_loss += loss

        valid_loss /= len(test_dloader)

    ckpt = {'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch}
    torch.save(ckpt, ckpt_path)

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

test_image_dir = 'test_images'
image_path_list = [os.path.join(test_image_dir, path) for path in os.listdir(test_image_dir)]

for image_path in image_path_list:
    inference(model, image_path, device, VOC_CLASSES)