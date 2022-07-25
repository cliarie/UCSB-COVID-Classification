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
    # fCP = open(f'analyze/fCP.txt', 'a+')
    # fCN = open(f'analyze/fCN.txt', 'a+')
    # fPN = open(f'analyze/fNP.txt', 'a+')
    # f = open(f'analyze/f.txt', 'a+')

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
#             TP += ((pred == 1) & (target.long()[:, 2].view_as(pred).data == 1)).cpu().sum()
#             TN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
# #             # FN    predict 0 label 1
#             FN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 1)).cpu().sum()
# #             # FP    predict 1 label 0
#             FP += ((pred == 1) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
#             print(TP,TN,FN,FP)
            FCN = ((pred == 0) & (target.long().view_as(pred) == 2)).cpu().sum() + ((pred == 2) & (target.long().view_as(pred) == 0)).cpu().sum()
            FCP = ((pred == 0) & (target.long().view_as(pred) == 1)).cpu().sum() + ((pred == 1) & (target.long().view_as(pred) == 0)).cpu().sum()
            FPN = ((pred == 1) & (target.long().view_as(pred) == 2)).cpu().sum() + ((pred == 2) & (target.long().view_as(pred) == 1)).cpu().sum()

            # if FCN:
            #     fCN.writelines("X "+batch_samples['name'][0].split('/')[6] + '\n')
            #     f.writelines("X "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fCP.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fPN.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            # elif FCP:
            #     fCP.writelines("X "+batch_samples['name'][0].split('/')[6] + '\n')
            #     f.writelines("X "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fCN.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fPN.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            # elif FPN:
            #     fPN.writelines("X "+batch_samples['name'][0].split('/')[6] + '\n')
            #     f.writelines("X "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fCP.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fCN.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            # else:
            #     fCN.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fCP.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            #     f.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            #     fPN.writelines("O "+batch_samples['name'][0].split('/')[6] + '\n')
            
#             print(output[:,1].cpu().numpy())
#             print((output[:,1]+output[:,0]).cpu().numpy())
#             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
    return targetlist, scorelist, predlist

import torchvision.models as models
model = models.resnet152(pretrained=True).cuda()
modelname = 'ResNet152'

device = 'cuda'

model.load_state_dict(torch.load('/home/claire/3ResNet152-overlay/model_backup/epoch150_ResNet152.pt'))
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

total_epoch = 1
for epoch in range(1, total_epoch+1):
    
    targetlist, scorelist, predlist = test(epoch)

    TC = ((predlist == 0) & (targetlist == 0)).sum()
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 2) & (targetlist == 2)).sum()

    FCp = ((predlist == 0) & (targetlist == 1)).sum() + ((predlist == 0) & (targetlist == 2)).sum()
    FPp = ((predlist == 1) & (targetlist == 0)).sum() + ((predlist == 1) & (targetlist == 2)).sum()
    FNp = ((predlist == 2) & (targetlist == 0)).sum() + ((predlist == 2) & (targetlist == 1)).sum()

    FCn = ((predlist == 1) & (targetlist == 0)).sum() + ((predlist == 2) & (targetlist == 0)).sum()
    FPn = ((predlist == 0) & (targetlist == 1)).sum() + ((predlist == 2) & (targetlist == 1)).sum()
    FNn = ((predlist == 0) & (targetlist == 2)).sum() + ((predlist == 1) & (targetlist == 2)).sum()

    print('TC=',TC,'TP=',TP,'TN=',TN,'FC',FCp,'FP=',FPp,'FN=',FNp)
    pC = TC / (TC + FCp)
    pP = TP / (TP + FPp)
    pN = TN / (TN + FNp)
    print('COVID precision',pC)
    print('PNA precision',pP)
    print('Normal precision',pN)
    p = (pC + pP + pN)/3

    rC = TC / (TC + FCn)
    rP = TP / (TP + FPn)
    rN = TN / (TN + FNn)
    print('COVID recall',rC)
    print('PNA recall',rP)
    print('Normal recall',rN)
    r = (rC + rP + rN)/3

    acc = (TC + TP + TN) / (TC + TP + TN + FCp + FPp + FNp)
    print('acc',acc)
    
    #confusion matrix
    print("confusion matrix")
    print (((predlist == 0) & (targetlist == 0)).sum())
    print(((predlist == 1) & (targetlist == 0)).sum())
    print(((predlist == 2) & (targetlist == 0)).sum())
    print(((predlist == 0) & (targetlist == 1)).sum())
    print(((predlist == 1) & (targetlist == 1)).sum())
    print(((predlist == 2) & (targetlist == 1)).sum())
    print(((predlist == 0) & (targetlist == 2)).sum())
    print(((predlist == 1) & (targetlist == 2)).sum())
    print(((predlist == 2) & (targetlist == 2)).sum())
    
    
    f = open(f'model_result/{modelname}.txt', 'a+')
    f.write('precision, recall, acc= \n')
    f.writelines(str(p))
    f.writelines('\n')
    f.writelines(str(r))
    f.writelines('\n')
    f.writelines(str(acc))
    f.writelines('\n')
    f.close()

    epoch1=150

    vote_pred = np.zeros((1,testset.__len__()))
    vote_score = np.zeros(testset.__len__())
    print('vote_pred',vote_pred)
    print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f}, average accuracy: {:.4f}'.format(
    epoch1, r, p, acc))

    f = open(f'model_result/test_{modelname}.txt', 'a+')
    f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f}, average accuracy: {:.4f}'.format(
    epoch1, r, p, acc))
    f.close()
