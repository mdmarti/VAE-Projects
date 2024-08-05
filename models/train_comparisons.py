from tqdm import tqdm
import torch

def train(model,optimizer,loaders,nEpochs=100,save_freq=25):

    train_losses,test_losses,val_losses = [],[],[]
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        train_loss,optimizer = model.train_epoch(loaders['train'],optimizer)

        model.eval()
        val_loss = model.val_epoch(loaders['val'])

        if epoch % save_freq == 0:
            model.save(optimizer)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses,test_losses,model,optimizer


def predict(model,trajectory):

    model.eval()
    trajectory = torch.from_numpy(trajectory).type(torch.FloatTensor).to(model.device)
    x = trajectory[:-1]
    y = trajectory[1:]

    yhat,_ = model(x)

    return yhat.detach().cpu().numpy()