import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import pandas as pd
import random 
from shutil import copyfile
import albumentations as albu
from sklearn.metrics import roc_auc_score
import re
import albumentations as albu
from skimage.io import imread, imsave
import skimage
import glob
import imageio

torch.cuda.empty_cache()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# data augmentation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

batchsize=10
import nibabel as nib

def read_txt(txt_path):
    #print(txt_path)
    with open(txt_path) as f:
        lines = f.readlines()
    #print(lines)
    txt_data = [line.strip() for line in lines]
    return txt_data

# dataloader
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_PNA, txt_Normal, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_PNA, txt_Normal]
        self.classes = ['COVID', 'PNA', 'Normal']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            for item in read_txt(self.txt_path[c]):
                for img in glob.iglob(os.path.join(self.root_dir,self.classes[c], item + "*.png")):
                    if (img == '/'): continue
                    cls_list = [img, c]
                    self.img_list.append(cls_list)
                    #print(img)
            
            #cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            #self.img_list += cls_list
        #print(self.img_list)
        #print(self.img_list[1][0])
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print(idx)
        img_path = self.img_list[idx][0]
        # print(f"path: {img_path}")
        # print(os.path.abspath("5MZST6Z8_Covid_slice_40.png"))
        if (not os.path.exists(img_path)): 
            print(f"bad path: {img_path}")
            return
        data = imageio.imread(img_path)
        #data = image.get_fdata()
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        image = Image.fromarray(rescaled).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1]),
                  'name' : str(self.img_list[idx][0])}
        return sample

if __name__ == '__main__':
    trainset = CovidCTDataset(root_dir='/home/claire/data/processedoverlay2/',
                              txt_COVID='/home/claire/data/data-split/COVID/train_COVID.txt',
                              txt_PNA='/home/claire/data/data-split/PNA/train_PNA.txt',
                              txt_Normal='/home/claire/data/data-split/Normal/train_Normal.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='/home/claire/data/processedoverlay2/',
                              txt_COVID='/home/claire/data/data-split/COVID/val_COVID.txt',
                              txt_PNA='/home/claire/data/data-split/PNA/val_PNA.txt',
                              txt_Normal='/home/claire/data/data-split/Normal/val_Normal.txt',
                              transform= val_transformer)
    
    testset = CovidCTDataset(root_dir='/home/claire/data/processedoverlay2/',
                              txt_COVID='/home/claire/data/data-split/COVID/test_COVID.txt',
                              txt_PNA='/home/claire/data/data-split/PNA/test_PNA.txt',
                              txt_Normal='/home/claire/data/data-split/Normal/test_Normal.txt',
                              transform= val_transformer)
    # print(trainset.__len__())
    # print(valset.__len__())
    # print(testset.__len__())

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=1, drop_last=False, shuffle=False)
    list_images = trainset.img_list
    #print(list_images)

alpha = None
device = 'cuda'

# training model (do not touch, should run right)
def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        
        # move data to device
        data, target, name = batch_samples['img'].to(device), batch_samples['label'].to(device), batch_samples['name']
#         print(target)
#         print(name)
        optimizer.zero_grad()
        output = model(data)
        
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
#         loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)
        train_loss += criteria(output, target.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f = open('/home/claire/3ResNet152-overlay/model_result/{}.txt'.format(modelname), 'w')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()
    return train_loss.cpu().detach().numpy()

# validation, take data and run it (should run without errors)
def val(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
    results = []
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
#            data = data[:, 0, :, :]
#            data = data[:, None, :, :]
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
#             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            
#             print(output[:,1].cpu().numpy())
#             print((output[:,1]+output[:,0]).cpu().numpy())
#             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
          
    return targetlist, scorelist, predlist, test_loss.cpu().detach().numpy()

# part to experiment on
# pretrained on ImageNet (classify cats n dogs) dataset bc not enough dataset (concept of transfer images)
# using resnet bc practical, skip so not overfit
import torchvision.models as models
model = models.resnet152(pretrained=True).cuda()
modelname = 'ResNet152'

# model.load_state_dict(torch.load('/home/pkao/covid/UCSB_COVID/model_backup/epoch25_ResNet50.pt'))
from numpy import *
import math
import matplotlib.pyplot as plt
# train
bs = 10
votenum = 10
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
# hyper-parameter tuning: change optimizer, experiment
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#scheduler = StepLR(optimizer, step_size=1)

# experimenting, so can just do 1 epoch to see if it runs
# will run 100 epochs (standard)
#total_epoch = 15
total_epoch = 150
train_loss_list = []
val_loss_list = []
for epoch in range(1, total_epoch+1):
    train_loss_list.append(train(optimizer, epoch))    
    #targetlist: ground truth; predlist: predictions
    targetlist, scorelist, predlist, val_loss = val(epoch)
    val_loss_list.append(val_loss)
#     print('target',targetlist)
#     print('score',scorelist)
#     print('predict',predlist)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
#     print(predlist)

    if epoch % votenum == 0:
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1
        vote_score = vote_score/votenum
        
        print('vote_pred', vote_pred)
        print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        
        
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        #accuracy, what we are comparing at the end
        print('acc',acc)
    #         AUC = roc_auc_score(targetlist, vote_score)
    #         print('AUCp', roc_auc_score(targetlist, vote_pred))
    #         print('AUC', AUC)
        
        
        # method to save epoch
        print(epoch)
        if epoch%1 == 0:
            if not os.path.exists("/home/claire/3ResNet152-overlay/model_backup/epoch{}_{}.pt".format(epoch,modelname)): open("/home/claire/3ResNet152-overlay/model_backup/epoch{}_{}.pt".format(epoch,modelname), "x")
            torch.save(model.state_dict(), "model_backup/epoch{}_{}.pt".format(epoch,modelname))  

            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}'.format(
            epoch, r, p, F1, acc))


            f = open('model_result/epoch{}_{}.txt'.format(modelname, epoch), 'w')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}'.format(
            epoch, r, p, F1, acc))
            f.close()
            
            t = [i for i in range(len(val_loss_list))]
            print(t)
            print(type(t))
            plt.plot(t, train_loss_list, 'r', label = "train") 
            plt.plot(t, val_loss_list, 'b', label = "validation") 
            plt.legend()
            plt.title('Loss vs Epochs')
            plt.savefig(os.path.join('model_result/', str(epoch) + 'ResNet152.png'))