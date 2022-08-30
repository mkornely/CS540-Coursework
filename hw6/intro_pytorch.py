import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Code adapted from Pytorch Tutorial of Training a Classifier :
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

def get_data_loader(training = True):
   
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training: 
        data_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    else:
        data_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    
    loader = torch.utils.data.DataLoader(data_set, batch_size = 64)
    
    return loader

def build_model():
   
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10))

    return model



def train_model(model, train_loader, criterion, T):
  
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0       
        for i, data in enumerate(train_loader, 0):            
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
        average=round(running_loss/i,3)
        percentage=round(100 *(correct/total),2)
        
        print("Train Epoch: " + str(epoch)+ "    Accuracy: "+ str(correct) +"/" + str(total)+ "("+ str(percentage)+ "%)     Loss: "+ str(average))



def evaluate_model(model, test_loader, criterion, show_loss = True):
   
    correct = 0
    total = 0
    running_loss=0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    percentage=round(100 *(correct/total),2)
    print("Accuracy: "+str(percentage)+ "%")
    

    if show_loss:
        average=round(running_loss/len(test_loader.dataset),4)
        print("Average Loss: "+str(average))
    


def predict_label(model, test_images, index):
    class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt"
                ,"Sneaker","Bag","Ankle Boot"]
    logits=model(test_images[index])
    prob = F.softmax(logits, dim=1)*100
    
    values,indicies= torch.topk(prob,3)
    
    for i in range(3): 
        print(str(class_names[indicies[0][i]])+": "+str(round(values[0][i].item(),2))+"%" )
        
    


if __name__ == '__main__':
    pass