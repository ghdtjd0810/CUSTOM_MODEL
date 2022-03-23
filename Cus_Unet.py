

import torch
from cmath import inf
import torch
import glob
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os


class Cus_Unet(nn.Module):
    def __init__(self, chkpt_dir="models_cus_unet"):
        super(Cus_Unet,self).__init__()
        self.file = os.path.join(chkpt_dir, "Cus_Unet")
        self.droput = nn.Dropout(0.2)
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding=1),
        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1),
        nn.ReLU(), # 여기까지는 괜찮음. 
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size = 2, stride = 2) # 여기서부터 확 줄어드는거임
        ) #

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding =1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, padding = 1)
        )


        ## 언풀 해서 컨캣해줄 함수.
        self.unpool_layer1= nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding=1),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() # 여기서부터 확 줄어드는거임
        ) #

        self.unpool_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.unpool_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding =1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.unpool_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, padding = 1)
        )




        self.upsampling1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2),
        )

        self.up_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size =3 , padding = 1 ),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.upsampling2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        )

        self.up_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.upsampling3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        )

        self.up_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsampling4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding =1 ),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels = 64, out_channels = 5, kernel_size = 1)
        )
        

        

    def forward(self,x):
        x1 = self.layer1(x) # 
        x2 = self.layer2(x1) #128
        x3 = self.layer3(x2) #256
        x4 = self.layer4(x3) #512
        x5 = self.layer5(x4)

        unx1 = self.unpool_layer1(x) #64
        unx2 = self.unpool_layer2(x1) # 128
        unx3 = self.unpool_layer3(x2) # 256
        unx4 = self.unpool_layer4(x3) # 512

        up1 = self.upsampling1(x5)
        up1_concat = torch.cat((unx4,up1), dim = 1)
        x6 = self.up_layer1(up1_concat)

        up2 = self.upsampling2(x6)
        up2_concat = torch.cat((unx3,up2), dim = 1)
        x7 = self.up_layer2(up2_concat)

        up3 = self.upsampling3(x7)
        up3_concat = torch.cat((unx2, up3), dim = 1)
        x8 = self.up_layer3(up3_concat)

        x9 = self.upsampling4(x8)
        up4_concat = torch.cat((unx1,x9),dim = 1)
        x9 = self.last_layer(up4_concat)
        
        return x9        
        
        
        
        
    def save(self):
        if isinstance(self, nn.DataParallel):
            torch.save(self.module.state_dict(), self.file)
        else:
            torch.save(self.state_dict(), self.file)

    def load(self):
        if isinstance(self, nn.DataParallel):
            self.module.load_state_dict(torch.load(self.file))
        else:
            self.load_state_dict(torch.load(self.file))