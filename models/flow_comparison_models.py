import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np



class circlePredictor(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,save_dir='./circle_model',loss_fn = nn.MSELoss(reduction='mean')):

        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.output = nn.Linear(hidden_size,input_size)

        self.save_dir=save_dir
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.loss_fn = loss_fn

        self.epoch = 0
        if torch.cuda.is_available(): 
            self.device='cuda' 
            self.to(torch.device('cuda'))
        else:
            self.device='cpu'
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir,'runs'))
        
    def forward(self,x):

        output,(h_n,c_n) = self.lstm(x)

        output = self.output(output)

        return output,(h_n,c_n)

    def train_epoch(self, loader, optimizer):

        optimizer.zero_grad()
        tl = []
        for ii,batch in enumerate(loader,start=self.epoch*len(loader)):

            data,dt= batch[:-1],batch[-1]
            #print(batch)
            x,y = torch.stack(data[:-1],axis=1).to(self.device),torch.stack(data[1:],axis=1).to(self.device)
            out,(h_n,c_n) = self.forward(x)

            train_loss = self.loss_fn(out,y)
            train_loss.backward()
            optimizer.step()
            self.writer.add_scalar('Train/loss',train_loss.item(),ii)
            tl.append(train_loss.item())

        return tl,optimizer

    def val_epoch(self,loader):

        vl = []
        with torch.no_grad():

            for batch in loader:
                data,dt= batch[:-1],batch[-1]
                
                x,y = torch.stack(data[:-1],axis=1).to(self.device),torch.stack(data[1:],axis=1).to(self.device)
                out,(h_n,c_n) = self.forward(x)
    
                val_loss = self.loss_fn(out,y)
                vl.append(val_loss.item())
            self.writer.add_scalar('Val/loss',np.nanmean(vl),self.epoch)

        return vl

    def save(self,optimizer):

        state_dict_full = {'model_dict':self.state_dict(),'opt_dict':optimizer.state_dict(),'epoch':self.epoch}
        torch.save(state_dict_full,os.path.join(self.save_dir,f'checkpoint_{self.epoch}.tar'))

    def load(self,path,optimizer):

        state_dict = torch.load(path)
        optimizer.load_state_dict(state_dict['opt_dict'])
        self.load_state_dict(state_dict['model_dict'])
        self.epoch = state_dict['epoch']


            
            
        