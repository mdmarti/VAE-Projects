import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import inspect

class latentSDE(object):
    
	def __init__(self,dim,diag_covar=True):
		self.dim,self.diag_covar = dim,diag_covar

	def predict(self,data):
		pass 
	
	def loss(self,data):
		pass

	def forward(self,data):
		pass
	def save(self):
		pass

	def load(self):
		pass 