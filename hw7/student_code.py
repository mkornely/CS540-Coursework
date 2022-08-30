# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.linear1=nn.Linear(400, 256)
        self.linear2=nn.Linear(256,128)
        self.linear3=nn.Linear(128,num_classes)
        self.flatten= nn.Flatten()
        self.relu=nn.ReLU()
        self.input_shape=input_shape

    def forward(self, x):
        shape_dict = {}
        
        out=self.conv1(x)
        out=self.relu(out)
        out=self.pool(out)
        shape_dict[1]=list(out.shape)
        
        out=self.conv2(out)
        out=self.relu(out)
        out=self.pool(out)
        shape_dict[2]=list(out.shape)
        
    
        out= self.flatten(out)
        shape_dict[3]=list(out.shape)

        out=self.linear1(out)
        out=self.relu(out)
        shape_dict[4]=list(out.shape)

        out=self.linear2(out)
        out=self.relu(out)
        shape_dict[5]=list(out.shape)

        out=self.linear3(out)
        shape_dict[6]=list(out.shape)

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    total_params=0
    for name, param in model.named_parameters():
        total_params+=param.numel()
    return total_params/1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

if __name__=="__main__":
    count_model_params()
