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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    # 
def test(epoch):
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
        predlist=[]
        scorelist=[]
        targetlist=[]
        scanlist=[]

        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
#            data = data[:, 0, :, :]
#            data = data[:, None, :, :]
#             print(target)
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
#             print(score[0][0]*100, score[0][1]*100, score[0][2]*100)
#             print(score.shape)
#             print(score[:3]*100)
            pred = output.argmax(dim=1, keepdim=True)
#             print(score[0][pred]*100)
#             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
            scanlist.append(batch_samples['name'][0].split('/')[6])
           
    return targetlist, scorelist, predlist, scanlist

import torchvision.models as models
model = models.resnet152(pretrained=True).cuda()
modelname = 'ResNet152'

device = 'cuda'

model.load_state_dict(torch.load('/home/claire/3ResNet152-overlay/model_backup/epoch20_ResNet152.pt'))
# test
bs = 10
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
vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

targetlist, scorelist, predlist, scanlist = test(1)


total = 0.0
correct = 0.0
covid=0.0
pna=0.0
normal=0.0
for i in range(len(scanlist) + 1):
    if (i>=0 and i != len(scanlist) and scanlist[i].split('_slice_')[0] == scanlist[i-1].split('_slice_')[0]):
        if predlist[i] == 0:
            covid+=1
        elif predlist[i] == 1:
            pna+=1
        else:
            normal += 1
    else:
        #evaluate, compare with target
        if i != 0:
            majority = max(covid, pna, normal)
            print(majority, covid, pna, normal, targetlist[i-1])
            if normal == majority and 2.0 == targetlist[i-1]:
                correct += 1
            elif pna == majority and 1.0 == targetlist[i-1]:
                correct += 1
            elif covid == majority and 0.0 == targetlist[i-1]:
                correct += 1
            total += 1
            if i == len(scanlist):
                break
                
        #reset, for the next scan
        covid = 0
        pna = 0
        normal = 0
        if predlist[i] == 0:
            covid+=1
        elif predlist[i] == 1:
            pna+=1
        else:
            normal += 1

print(correct,total)

