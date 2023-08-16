import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torch import nn
import torch
from  lib.networks.iteration import iteration
from losses import  FocalLoss
from lib.networks.isegformer import Segformer as isegformer
from lib.networks.interCNN import interCNN as interCNN
from lib.networks.isnet import Isnet as isnet
from lib.networks.PRNet import PRNet as PRNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='isnet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = "./results/isnet/"

iter_num = 5
learning_rate = 0.0001
imgsize = 512
epochs = 800
save_freq = 10
train_dataset_path = r"/data1/wjw/418/MoNuSeg_split/train"
val_dataset_path = r"/data1/wjw/418/MoNuSeg_split/test"
batch_size = 1
mask_ratio = 0.5
dataset_name = "TNBC"
modelname = "isegformer"
method = "lines"
use_sim = False
npy_file_path = r"/data1/wjw/418/MoNuSeg_splitnpy/"
patchsize = 32
diameter = 0

if modelname == "isnet":
    use_sim = True

if method == "click" or method == "disk":
    diameter = 15
elif method == "localGT" or  method == "localeGT":
    diameter = 30

print(modelname,method,diameter)

from utils import JointTransform2D, ImageToImage2D, Image2D
imgchant = 6

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1), long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(train_dataset_path,dataset_name, tf_train)
val_dataset = ImageToImage2D(val_dataset_path,dataset_name, tf_val)
dataloader = DataLoader(train_dataset,drop_last=True, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset,drop_last=True, batch_size=batch_size, shuffle=True)


if modelname == "isegformer":
    model = isegformer(channels = 6,decoder_dim = 512).cuda()
elif modelname == "interCNN":
    model = interCNN(in_channels=6,num_classes=2).cuda()
elif modelname == "isnet":
    model = isnet(channels = 6,decoder_dim = 512).cuda()
elif modelname == "PRNet":
    model = PRNet(c_in = 6,c_blk=128,n_classes = 2).cuda()


model = nn.DataParallel(model)

Dice_Loss = FocalLoss(2)
# optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,weight_decay=0.0001)
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

print(modelname,batch_size,use_sim)

for epoch in range(epochs):
    model.train()
    epoch_running_loss = 0
    epoch_dc = 0
    epoch_jc = 0

    for batch_idx, (X_batch, y_batch, real_batch, img_name) in enumerate(dataloader):
        # X_batch, y_batch = hint_add(X_batch, y_batch)
        X_batch = Variable(X_batch.cuda())
        y_batch = Variable(y_batch.cuda())
        loss,dc,jc,output,tmp,tmp2 = iteration(X_batch, y_batch,real_batch,iter_num,model,optimizer,img_name,npy_file_path,patchsize,epoch=epoch,
                                                 train=True,method = method ,diameter=diameter,use_sim=use_sim)
        epoch_running_loss += loss.item()
        epoch_dc +=dc
        epoch_jc+=jc

    # ===================log========================
    train_len = len(dataloader)
    print('epoch [{}/{}], loss:{:.4f},dice:{:.4f},jc:{:.4f}'
          .format(epoch, epochs, epoch_running_loss / (train_len),epoch_dc / (train_len),epoch_jc / (train_len)))


    if (epoch % save_freq) == 0:
        total_dc = 0
        total_loss = 0
        total_jc = 0
        for batch_idx, (X_batch, y_batch, real_batch, img_name) in enumerate(valloader):

            # print(batch_idx)
            if isinstance(img_name[0], str):
                image_filename = img_name[0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
            # X_batch, y_batch = hint_add(X_batch, y_batch)
            X_batch = Variable(X_batch.cuda())
            y_batch = Variable(y_batch.cuda())
            loss, dc, jc,output,tmp,tmp2 = iteration(X_batch, y_batch,real_batch,iter_num,model,optimizer, img_name,npy_file_path,patchsize,epoch=epoch,
                                                 train=False,method = method ,diameter=diameter,use_sim=use_sim )
            total_dc += dc
            total_loss +=loss
            total_jc += jc
            yHaT = tmp
            yval = tmp2
            del X_batch, y_batch, output,tmp, tmp2
            yHaT[yHaT == 1] = 255
            yval[yval == 1] = 255
            fulldir = direc + "/{}/".format(epoch)
            # print(fulldir+image_filename)
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            if not image_filename.endswith(".png"):
                image_filename = image_filename + ".png"

            cv2.imwrite(fulldir + image_filename, yHaT[0, :, :])
            # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])

        print("--------------------------------------")
        val_len = len(valloader)

        print('epoch [{}/{}], loss:{:.4f},dice:{:.4f},jc:{:.4f}'
                  .format(epoch, epochs, total_loss / (val_len ),total_dc / (val_len ) ,total_jc / (val_len )))

        torch.save(model.state_dict(), fulldir + modelname + str(total_dc / (val_len ))+".pth")

        print("--------------------------------------")


