from torch.optim import Adam, lr_scheduler,SGD
from tqdm import tqdm 
#from data import toyDataset
import torchvision

def train(newNetwork,dataloaders,nEpochs,opt='adam',lr=5e-5,test_freq=5,save_freq=100,wd=0.,gamma=None):

    if opt=='adam':
        optimizer = Adam(newNetwork.parameters(), lr=lr,weight_decay=wd) # 8e-5
    else:
        optimizer = SGD(newNetwork.parameters(), lr=lr,weight_decay=wd) # 8e-5
    if gamma != None:
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=gamma)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=20,min_lr=1e-10)
    for epoch in tqdm(range(1,nEpochs + 1),desc='Training linear latent sde'):

        trainLoss,optimizer = newNetwork.train_epoch(dataloaders['train'],optimizer)
        if gamma != None:
            scheduler.step()
        if epoch % test_freq == 0:
            testLoss = newNetwork.test_epoch(dataloaders['test'])
            
            #Ds = newNetwork.D(next(iter(dataloaders['test']))).detach().cpu()
            #grid = torchvision.utils.make_grid(Ds[0:4,:],nrow=1)
            #newNetwork.writer.add_image('D matrix',grid,newNetwork.epoch)
        if epoch % save_freq == 0:
            
            newNetwork.save()

    print("Done training!")
    print(f"Final train loss: {trainLoss/len(dataloaders['train'])}")
    print(f"Final test loss: {testLoss/len(dataloaders['test'])}")
    return newNetwork

def trainAlternating(newNetwork,dataloaders,nEpochs,lr=5e-5,switch_freq = 2,test_freq=5,save_freq=100,wd=0.,gamma=None):

    mlpOptimizer = Adam(newNetwork.MLP.parameters(),lr=lr,weight_decay=wd)
    sigmaOptimizer = Adam(newNetwork.D.parameters(),lr=lr,weight_decay=wd)
    
    if gamma != None:
        mlpscheduler = lr_scheduler.ExponentialLR(optimizer=mlpOptimizer,gamma=gamma)
        sigmascheduler = lr_scheduler.ExponentialLR(optimizer=sigmaOptimizer,gamma=gamma)
    else:
        mlpscheduler = lr_scheduler.ReduceLROnPlateau(optimizer=mlpOptimizer,mode='min',factor=0.5,patience=20,min_lr=1e-10) 
        sigmascheduler = lr_scheduler.ReduceLROnPlateau(optimizer=sigmaOptimizer,mode='min',factor=0.5,patience=20,min_lr=1e-10)    
    
    type = 'mu'
    for epoch in tqdm(range(1,nEpochs + 1),desc='Training linear latent sde'):

        if type == 'mu':
            trainLoss,mlpOptimizer = newNetwork.train_epoch(dataloaders['train'],mlpOptimizer)
            
            mlpscheduler.step()

            if epoch % switch_freq == 0:
                type == 'sigma'

        else:
            trainLoss,sigmaOptimizer = newNetwork.train_epoch(dataloaders['train'],sigmaOptimizer)
            
            sigmascheduler.step()

            if epoch % switch_freq == 0:
                type == 'mu'
           
        
        if epoch % test_freq == 0:
            testLoss = newNetwork.test_epoch(dataloaders['test'])
            
            #Ds = newNetwork.D(next(iter(dataloaders['test']))).detach().cpu()
            #grid = torchvision.utils.make_grid(Ds[0:4,:],nrow=1)
            #newNetwork.writer.add_image('D matrix',grid,newNetwork.epoch)
        if epoch % save_freq == 0:
            
            newNetwork.save()

    print("Done training!")
    print(f"Final train loss: {trainLoss/len(dataloaders['train'])}")
    print(f"Final test loss: {testLoss/len(dataloaders['test'])}")
    return newNetwork

