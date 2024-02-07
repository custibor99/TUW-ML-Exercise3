from CustomImageDataset import CustomImageDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from Models import SimpleNetwork
from torch.optim import Adam, Optimizer
from torch import nn
import torch
from datetime import datetime

class ModelTrainer:
    def __init__(self, model:nn.Module, train_dataloader:DataLoader, test_dataloader:DataLoader, loss_fn, optimizer:Optimizer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for data in tqdm(iter(self.train_dataloader)):
            inputs, outputs = data

            #Zero gradient every batch
            self.optimizer.zero_grad()

            #Get predictions
            predictions = self.model(inputs)

            #COmpute loss and gradient
            loss = self.loss_fn(predictions, outputs)
            loss.backward()

            #Adjust weights
            self.optimizer.step()




            
        return last_loss
    
    def train_n_epochs(self, n_epochs):
        epoch_number = 0
        best_vloss = 1_000_000
        
        for epoch in range(n_epochs):
            print('EPOCH {}:'.format(epoch + 1))
            self.model.train(True)
            avg_loss = self.train_one_epoch(None, None)
            running_vloss = 0.0
            self.model.eval()

            #Possible validation processing
            #with torch.no_grad():
            #    for i, vdata in enumerate(validation_loader):
            #    vinputs, vlabels = vdata
            #    voutputs = model(vinputs)
            #    vloss = loss_fn(voutputs, vlabels)
            #    running_vloss += vloss
            
            print('LOSS train {}'.format(avg_loss))

            #save best model based on validation
            #if avg_vloss < best_vloss:
            #best_vloss = avg_vloss
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            #torch.save(model.state_dict(), model_path)



train_dataset = CustomImageDataset("../data/train/", n_images=50)
train_dataloader = DataLoader(train_dataset, batch_size=2)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


model = SimpleNetwork()
trainer = ModelTrainer(model, train_dataloader, None, nn.MSELoss(), Adam(model.parameters()))
trainer.train_n_epochs(5)