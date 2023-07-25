from torch.optim import Adam, lr_scheduler
from tqdm import tqdm 
from data import toyDataset
import torchvision

def train(newNetwork,dataloaders,nEpochs,lr=5e-5,test_freq=5,save_freq=100,wd=0.,step_size=100):

    optimizer = Adam(newNetwork.parameters(), lr=lr,weight_decay=wd) # 8e-5
    if step_size != None:
        scheduler = lr_scheduler.StepLR(optimizer=optimizer,step_size=step_size,gamma=0.9)
    for epoch in tqdm(range(1,nEpochs + 1),desc='Training linear latent sde'):

        trainLoss,optimizer = newNetwork.train_epoch(dataloaders['train'],optimizer)
        if step_size != None:
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

def trainAlternating(newNetwork,dataloaders,nEpochs,lr=5e-5,test_freq=5,save_freq=100,wd=0.,step_size=100):

    mlpOptimizer = Adam(newNetwork.MLP.parameters(),lr=lr,weight_decay=wd)
    sigmaOptimizer = Adam(newNetwork.D.parameters(),lr=lr,weight_decay=wd)
    
    if step_size != None:
        mlpscheduler = lr_scheduler.StepLR(optimizer=mlpOptimizer,step_size=step_size//2,gamma=0.9)
        sigmascheduler = lr_scheduler.StepLR(optimizer=sigmaOptimizer,step_size=step_size//2,gamma=0.9)
    for epoch in tqdm(range(1,nEpochs + 1),desc='Training linear latent sde'):

        if epoch % 2 == 0:
            trainLoss,mlpOptimizer = newNetwork.train_epoch(dataloaders['train'],mlpOptimizer)
            if step_size != None:
                mlpscheduler.step()

        else:
            trainLoss,sigmaOptimizer = newNetwork.train_epoch(dataloaders['train'],sigmaOptimizer)
            if step_size != None:
                sigmascheduler.step()
           
        
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

