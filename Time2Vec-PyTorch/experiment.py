from Data import ToyDataset
from gunpointData import GunPointData
from periodic_activations import SineActivation, CosineActivation
import torch
import pyts
from torch.utils.data import DataLoader
from Pipeline import AbstractPipelineClass
from torch import nn
from Model import Model
import numpy as np
import matplotlib.pyplot as plt


class ToyPipeline(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model
        #self.data_train, self.data_test, self.target_train, self.target_test = load_gunpoint(return_X_y=True)
        #print("printing data", self.data_test, self.target_test)


    
    def train(self):
        loss_fn = nn.CrossEntropyLoss()

        dataset = ToyDataset()
        #dataset = GunPointData()
        #print(dataset)

        dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 25
        epoch_loss = 0
        count = 0
        running_acc = 0
        acc_values = []
        for ep in range(num_epochs):

            for x, y in dataloader:
                optimizer.zero_grad()

                y_pred = self.model(x.unsqueeze(1).float())
                ## take the argmax of the predictions on the first axis

                argmax = torch.max(y_pred, 1)

                acc = torch.sum(argmax[1] == y)/float(len(y))
                running_acc += acc/1.3
                #print("labels", argmax[1], y)
                #print("acc",acc)
                #print("shapes", y_pred.shape, y.shape)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                count+=1

                #print("epoch: {}, loss:{}".format(ep, loss.item()))
            print("epoch_loss", (epoch_loss/count))
            acc_values.append(running_acc/count)
            #print("acc", running_acc/count)

        #plt.plot(np.array(acc_values), 'r')
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x

if __name__ == "__main__":
     pipe = ToyPipeline(Model("sin", 12))
     pipe.train()

    #pipe = ToyPipeline(Model("cos", 100))
    #pipe.train()
