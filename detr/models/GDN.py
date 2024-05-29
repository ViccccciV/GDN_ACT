import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils

import numpy as np



# class GDN(nn.Module):
#     def __init__(self, input_size=(256,256), img_nums=1, input_c=3, lin_hidden_dims=(512, 16), out_size=64):
#         super(GDN,self).__init__()
#         self.input_size = input_size
#         self.input_c = input_c
#         self.img_nums = img_nums
#         self.out_size = out_size

#         self.size_after_conv = (256, 37, 25)
#         self.lin_hidden_dims = lin_hidden_dims

#         #encoder
#         self.f1 = nn.Conv2d(input_c,32,kernel_size=4,padding=1,stride=2)
#         self.b1 = nn.BatchNorm2d(32)
#         self.f2 = nn.Conv2d(32,64,kernel_size=4,padding=1,stride=2)
#         self.b2 = nn.BatchNorm2d(64)
#         self.f3 = nn.Conv2d(64,128,kernel_size=4,padding=1,stride=2)
#         self.b3 = nn.BatchNorm2d(128)
#         self.f4 = nn.Conv2d(128,256,kernel_size=4,padding=1,stride=2)
#         self.b4 = nn.BatchNorm2d(256)
#         self.f5 = nn.Conv2d(256,512,kernel_size=4,padding=1,stride=2)
#         self.b4 = nn.BatchNorm2d(512)

#         self.lin_layers = nn.ModuleList()
#         self.lin_layers.append(nn.Linear(np.prod(self.size_after_conv), self.lin_hidden_dims[0]))
#         self.lin_layers.append(nn.LeakyReLU())

#         for i in range(len(self.lin_hidden_dims)-1):
#           self.lin_layers.append(nn.Linear(self.lin_hidden_dims[i], self.lin_hidden_dims[i+1]))
#           self.lin_layers.append(nn.LeakyReLU())

#         self.lin_layers.append(nn.Linear(self.lin_hidden_dims[-1], self.out_size))
#         self.lin_layers.append(nn.Sigmoid())


#     def encoder(self,input):
#         x = F.leaky_relu(self.f1(input))
#         x = self.b1(x)
#         x = F.leaky_relu(self.f2(x))
#         x = self.b2(x)
#         x = F.leaky_relu(self.f3(x))
#         x = self.b3(x)
#         x = F.leaky_relu(self.f4(x))
#         x = self.b4(x)
#         x = F.leaky_relu(self.f5(x))
#         x = self.b5(x)
#         #print(x.shape)

#         x = torch.flatten(x, start_dim=1)

#         for layer in self.lin_layers:
#             x = layer(x)
#         return x


#     def forward(self,input):
#         return self.encoder(input)


class GDN(nn.Module):
    def __init__(self, input_size=(256,256), img_nums=1, input_c=3, lin_hidden_dims=(512, 16), out_size=64):
        super(GDN,self).__init__()
        self.input_size = input_size
        self.input_c = input_c
        self.img_nums = img_nums
        self.out_size = out_size

        # uncropped
        # self.size_after_conv = (512, 18, 12)

        # cropped
        self.size_after_conv = (512, 18, 9)
        
        self.lin_hidden_dims = lin_hidden_dims

        #encoder
        self.f1 = nn.Conv2d(input_c,32,kernel_size=4,padding=1,stride=2)
        self.b1 = nn.BatchNorm2d(32)
        self.f2 = nn.Conv2d(32,64,kernel_size=4,padding=1,stride=2)
        self.b2 = nn.BatchNorm2d(64)
        self.f3 = nn.Conv2d(64,128,kernel_size=4,padding=1,stride=2)
        self.b3 = nn.BatchNorm2d(128)
        self.f4 = nn.Conv2d(128,256,kernel_size=4,padding=1,stride=2)
        self.b4 = nn.BatchNorm2d(256)
        self.f5 = nn.Conv2d(256,512,kernel_size=4,padding=1,stride=2)
        self.b5 = nn.BatchNorm2d(512)

        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(np.prod(self.size_after_conv), self.lin_hidden_dims[0]))
        self.lin_layers.append(nn.LeakyReLU())

        for i in range(len(self.lin_hidden_dims)-1):
          self.lin_layers.append(nn.Linear(self.lin_hidden_dims[i], self.lin_hidden_dims[i+1]))
          self.lin_layers.append(nn.LeakyReLU())

        self.lin_layers.append(nn.Linear(self.lin_hidden_dims[-1], self.out_size))
        self.lin_layers.append(nn.Sigmoid())


    def encoder(self,input):
        x = F.leaky_relu(self.f1(input))
        x = self.b1(x)
        x = F.leaky_relu(self.f2(x))
        x = self.b2(x)
        x = F.leaky_relu(self.f3(x))
        x = self.b3(x)
        x = F.leaky_relu(self.f4(x))
        x = self.b4(x)
        x = F.leaky_relu(self.f5(x))
        x = self.b5(x)
        # print(x.shape)

        x = torch.flatten(x, start_dim=1)

        for layer in self.lin_layers:
            x = layer(x)
        return x


    def forward(self,input):
        return self.encoder(input)