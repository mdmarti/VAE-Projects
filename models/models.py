import torch
import numpy as np
from torch import nn
import torch.Functional as F
from torch.distributions import LowRankMultivariateNormal

class encoder(nn.Module):

    def __init__(self,z_dim):

        """
        encoder for birdsong VAEs
        """

        super(encoder,self).__init__()

        self.z_dim = z_dim

        self.encoder_conv = nn.Sequential(nn.BatchNorm2d(1),
                                nn.Conv2d(1, 8, 3,1,padding=1),
                                nn.Relu(),
                                nn.BatchNorm2d(8),
                                nn.Conv2d(8, 8, 3,2,padding=1),
                                nn.Relu(),
                                nn.BatchNorm2d(8),
                                nn.Conv2d(8, 16,3,1,padding=1),
                                nn.Relu(),
                                nn.BatchNorm2d(16),
                                nn.Conv2d(16,16,3,2,padding=1),
                                nn.Relu(),
                                nn.BatchNorm2d(16),
                                nn.Conv2d(16,24,3,1,padding=1),
                                nn.Relu(),
                                nn.BatchNorm2d(24),
                                nn.Conv2d(24,24,3,2,padding=1),
                                nn.Relu(),
                                nn.BatchNorm2d(24),
                                nn.Conv2d(24,32,3,1,padding=1),
                                nn.Relu())

        self.encoder_fc = nn.Sequential(nn.Linear(8192,1024),
                                nn.Relu(),
                                nn.Linear(1024,256),
                                nn.Relu())

        self.fc11 = nn.Linear(256,64)
        self.fc12 = nn.Linear(256,64)
        self.fc13 = nn.Linear(256,64)
        self.fc21 = nn.Linear(64,self.z_dim)
        self.fc22 = nn.Linear(64,self.z_dim)
        self.fc23 = nn.Linear(64,self.z_dim)
        
    def encode(self,x):

        h = self.encoder_conv(x)
        h = h.view(-1,8192)
        h = torch.cat(h,torch.zeros(h.shape[0],1,device=h.device))
        h = self.encoder_fc(x)
        mu = F.relu(self.fc11(h))
        u = F.relu(self.fc12(h))
        d = F.relu(self.fc13(h))
        mu = self.fc21(h)
        u = self.fc22(h)
        d = self.fc23(h)
        
        return mu, u.unsqueeze(-1),d.exp()

    def encode_with_time(self,x,encode_times):

        h = self.encoder_conv(x)
        h = h.view(-1,8192)
        h = torch.cat(h,encode_times)
        h = self.encoder_fc(x)
        mu = F.relu(self.fc11(h))
        u = F.relu(self.fc12(h))
        d = F.relu(self.fc13(h))
        mu = self.fc21(h)
        u = self.fc22(h)
        d = self.fc23(h)
        
        return mu, u.unsqueeze(-1),d.exp()

    def sample_z(self,mu,u,d):

        dist = LowRankMultivariateNormal(mu,u,d)

        z_hat = dist.rsample()

        return z_hat




