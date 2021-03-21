# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:56:12 2021

@author: Sai Mudumba
"""

import argparse
import numpy as np
import torch
import os
import torchvision.transforms as tvt
import glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="HW04 Training/Validation")
parser.add_argument("--root_path", required=True, type=str)
parser.add_argument("--class_list", required=True, nargs="*", type=str)
args, args_other = parser.parse_known_args()

root_path = args.root_path
class_list = args.class_list



confusion_matrix = torch.zeros(len(class_list), len(class_list))

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(123008, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 123008)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(25088, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 25088)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding = 1)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(28800, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 28800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


path_saved_model = "./net1"
net = Net1()
if torch.cuda.is_available():
    net.cuda()
net.load_state_dict(torch.load(path_saved_model))
print(net)

net = net.eval()
filename_for_results = "classification_results_Net1" + str(7) + ".txt"
FILE = open(filename_for_results, 'w')

class dataset_class(Dataset):
    def __init__(self, imageroot, class_list, transform):
        """
        Make use of the arguments from argparse
        
        Initialize your program-defined variables 
        e.g., image path lists for each class 
        
        Could also maintain label_array
        0 -- refrigerator
        1 -- airplane
        2 -- giraffe
        3 -- cat
        4 -- elephant
        5 -- dog
        6 -- train
        7 -- horse
        8 -- boat
        9 -- truck
        
        Initialize the required transforms
        
        Load color image(s), apply necessary data conversion and transformation
        "/Users/Sai Mudumba/Documents/PS2/*/*/"
        e.g., if an image is loaded in H x W x C (Height x Width x Channels) format
        rearrange it in C x H x W format, normalize values from 0-255 to 0-1
        and apply the necessary transformation.
        
        Convert the corresponding label in 1-hot encoding
        
        """
        self.imageroot = imageroot
        self.class_list = class_list
        self.transform = transform
        
        
        self.x_filenames = glob.glob(self.imageroot + "**/*.jpg")
        self.x_data = [self.transform(Image.open(filename)) for filename in self.x_filenames]
        
            
        """ Here, define the image paths using the glob function, and label the images"""
        self.i = 0
        self.y_data = torch.zeros(len(self.x_filenames))
        
        for clist in self.class_list: # in the command call we specify a list of classes of interest for training, and this goes through each of that class
            for name in glob.iglob(self.imageroot + "/" + str(clist) + "/*.jpg", recursive=True): # this goes over each of the images in the folder
                
                if clist == "refrigerator":
                    self.y_data[self.i] = int(0)
                elif clist == "airplane":
                    self.y_data[self.i] = int(1)
                elif clist == "giraffe":
                    self.y_data[self.i] = int(2)
                elif clist == "cat":
                    self.y_data[self.i] = int(3)
                elif clist == "elephant":
                    self.y_data[self.i] = int(4)
                elif clist == "dog":
                    self.y_data[self.i] = int(5)
                elif clist == "train":
                    self.y_data[self.i] = int(6)
                elif clist == "horse":
                    self.y_data[self.i] = int(7)
                elif clist == "boat":
                    self.y_data[self.i] = int(8)
                elif clist == "truck":
                    self.y_data[self.i] = int(9)
                
                self.i += 1    

    def __len__(self):
        """
        return the total number of images
        refer to pytorch documentation for more details
        """
        return len(self.x_filenames)
        
    def __getitem__(self, index):
        """        
        Return the processed images and labels in 1-hot encoded format
        """        
        return self.x_data[index], self.y_data[index]


transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

epochs = 7
batch_size = 10
validation_dataset = dataset_class(root_path, class_list, transform)
validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

correct = 0
total = 0
class_correct = [0] * len(class_list)
class_total = [0] * len(class_list)

for i,data in enumerate(validation_data_loader):
    images, labels = data
    images = images.to(device)
    labels = labels.to(torch.long)
    labels = labels.to(device)
    
    outputs = net(images)
    
    ##  max() returns two things: the max value and its index in the 10 element
    ##  output vector.  We are only interested in the index --- since that is 
    ##  essentially the predicted class label:
    _, predicted = torch.max(outputs.data, 1)
    
    for label,prediction in zip(labels,predicted):
        confusion_matrix[label][prediction] += 1
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
    comp = predicted == labels       
    for j in range(batch_size):
        label = labels[j]
        ##  The following works because, in a numeric context, the boolean value
        ##  "False" is the same as number 0 and the boolean value True is the 
        ##  same as number 1. For that reason "4 + True" will evaluate to 5 and
        ##  "4 + False" will evaluate to 4.  Also, "1 == True" evaluates to "True"
        ##  "1 == False" evaluates to "False".  However, note that "1 is True" 
        ##  evaluates to "False" because the operator "is" does not provide a 
        ##  numeric context for "True". And so on.  In the statement that follows,
        ##  while  c[j].item() will either return "False" or "True", for the 
        ##  addition operator, Python will use the values 0 and 1 instead.
        class_correct[label] += comp[j].item()
        class_total[label] += 1
for j in range(len(class_list)):
    print('Prediction accuracy for %5s : %2d %%' % (class_list[j], 100 * class_correct[j] / class_total[j]))
    FILE.write('\n\nPrediction accuracy for %5s : %2d %%\n' % (class_list[j], 100 * class_correct[j] / class_total[j]))
print("\n\n\nOverall accuracy of the network on the 5000 validation images: %d %%" % (100 * correct / float(total)))
FILE.write("\n\n\nOverall accuracy of the network on the 5000 validation images: %d %%\n" % (100 * correct / float(total)))
print("\n\nDisplaying the confusion matrix:\n")
FILE.write("\n\nDisplaying the confusion matrix:\n\n")

out_str = "         "
for j in range(len(class_list)):  out_str +=  "%7s" % class_list[j] 
print(out_str + "\n")
FILE.write(out_str + "\n\n")
for i,label in enumerate(class_list):
    out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                              for j in range(len(class_list))]
    out_percents = ["%.2f" % item.item() for item in out_percents]
    out_str = "%6s:  " % class_list[i]
    for j in range(len(class_list)): out_str +=  "%7s" % out_percents[j]
    print(out_str)
    FILE.write(out_str + "\n")
FILE.close() 

import seaborn as sn
import pandas as pd
confusion_matrix = confusion_matrix.to(torch.long)
confusion_matrix = confusion_matrix.numpy()

print(confusion_matrix)   
 
'''https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea'''
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in class_list],
                  columns = [i for i in class_list])
plt.figure(dpi=300,figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='2')
plt.xlabel("Output Label")
plt.ylabel("True Label")
plt.title("Net 1 Confusion Matrix [Rows: True Label] [Columns: Output Labels]")
plt.savefig("net1_confusion_matrix.jpg")
plt.show()