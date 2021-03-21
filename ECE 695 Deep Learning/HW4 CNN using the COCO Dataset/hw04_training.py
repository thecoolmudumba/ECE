# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:09:09 2021

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

    # CHECK 1: Works
    # print(root_path)
    # print(class_list)

# Initialize random seed to 0 so that results are reproducible
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


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
train_dataset = dataset_class(root_path, class_list, transform)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


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
    
def run_code_for_training(net, number):
    # filename_for_out = "NET1_performance_numbers_" + str(epochs) + ".txt"
    # FILE = open(filename_for_out, 'w')
    # net = copy.deepcopy(net)
    
    
    dtype = torch.float64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    
    loss_tally = []
    
    print("\n\nStarting training loop...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(torch.long)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 500 == 0:
                print("\n[epoch: %d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                loss_tally.append(running_loss / float(500))
                # FILE.write("\n[epoch: %d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                running_loss = 0.0
        
    print("\nFinished Training\n")
    
    path_saved_model = "./net"+str(number)
    torch.save(net.state_dict(), path_saved_model)
    return loss_tally
    
model1 = Net1()
loss_tally_net1 = run_code_for_training(model1, 1)    

model2 = Net2()
loss_tally_net2 = run_code_for_training(model2, 2)   
    
model3 = Net3()
loss_tally_net3 = run_code_for_training(model3, 3)

plt.figure(figsize=(10,5))
plt.title("Labeling Loss vs. Iterations")
plt.plot(loss_tally_net1, 'g', label="Net1 Training Loss")
plt.plot(loss_tally_net2, 'b', label="Net2 Training Loss")
plt.plot(loss_tally_net3, 'r', label="Net3 Training Loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.savefig("train_loss.jpg")
plt.show()
    



# def run_code_for_training(self, net, display_images=True):        
#         filename_for_out = "performance_numbers_" + str(self.epochs) + ".txt"
#         FILE = open(filename_for_out, 'w')
#         net = copy.deepcopy(net)
#         net = net.to(self.device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum)
#         print("\n\nStarting training loop...")
#         start_time = time.perf_counter()
#         loss_tally = []
#         elapsed_time = 0.0
#         for epoch in range(self.epochs):  
#             print("")
#             running_loss = 0.0
#             for i, data in enumerate(self.train_data_loader):
#                 inputs, labels = data
                
#                 if i % 1000 == 999:
#                     current_time = time.perf_counter()
#                     elapsed_time = current_time - start_time 
#                     print("\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Ground Truth:     " % 
#                           (epoch+1, self.epochs, i+1, elapsed_time) + 
#                           ' '.join('%10s' % self.class_labels[labels[j]] for j in range(self.batch_size)))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 ##  Since PyTorch likes to construct dynamic computational graphs, we need to
#                 ##  zero out the previously calculated gradients for the learnable parameters:
#                 optimizer.zero_grad()
#                 outputs = net(inputs)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item()
#                 if i % 1000 == 999:
#                     _, predicted = torch.max(outputs.data, 1)
#                     print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Predicted Labels: " % 
#                      (epoch+1, self.epochs, i+1, elapsed_time ) +
#                      ' '.join('%10s' % self.class_labels[predicted[j]] for j in range(self.batch_size)))
#                     avg_loss = running_loss / float(2000)
#                     loss_tally.append(avg_loss)
#                     print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Loss: %.3f" % 
#                                                                    (epoch+1, self.epochs, i+1, elapsed_time, avg_loss))    
#                     FILE.write("%.3f\n" % avg_loss)
#                     FILE.flush()
#                     running_loss = 0.0
#                     if display_images:
#                         logger = logging.getLogger()
#                         old_level = logger.level
#                         logger.setLevel(100)
#                         plt.figure(figsize=[6,3])
#                         plt.imshow(np.transpose(torchvision.utils.make_grid(inputs, 
#                                                             normalize=False, padding=3, pad_value=255).cpu(), (1,2,0)))
#                         plt.show()
#                         logger.setLevel(old_level)
#                 loss.backward()
#                 optimizer.step()
#         print("\nFinished Training\n")
#         self.save_model(net)
#         plt.figure(figsize=(10,5))
#         plt.title("Labeling Loss vs. Iterations")
#         plt.plot(loss_tally)
#         plt.xlabel("iterations")
#         plt.ylabel("loss")
#         plt.legend()
#         plt.savefig("playing_with_skips_loss.png")
#         plt.show()
# # val_dataset = your_dataset_class(args.imagenet_root, args.class_list, False, transform)
# # val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)