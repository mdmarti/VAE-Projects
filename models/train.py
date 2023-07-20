from torch.optim import Adam, lr_scheduler
from tqdm import tqdm 
from data import toyDataset
import torchvision

def train(newNetwork,dataloaders,nEpochs,lr=1e-4,test_freq=5,save_freq=100,wd=0.):

    optimizer = Adam(newNetwork.parameters(), lr=lr,weight_decay=wd) # 8e-5
    #scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.999)
    for epoch in tqdm(range(1,nEpochs + 1),desc='Training linear latent sde'):

        trainLoss,optimizer = newNetwork.train_epoch(dataloaders['train'],optimizer)
        #scheduler.step()
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
