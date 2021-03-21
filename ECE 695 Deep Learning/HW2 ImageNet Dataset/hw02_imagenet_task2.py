# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:59:04 2021

@author: Sai Mudumba

"""

import argparse
import glob
import torch
from PIL import Image
import os
import numpy as np

"""
--imagenet_root corresponds to the top folder containing both Train and Val subfolders as created in Task 1. 
The following is an example call to this script
python hw02_imagenet_task2.py --imagenet_root <path_to_imagenet_root> --class_list 'cat' 'dog'

"""

parser = argparse.ArgumentParser(description='HW02 Task2')
parser.add_argument('--imagenet_root', type=str, required=True)
parser.add_argument('--class_list', nargs = '*', type=str, required=True)
args, args_other = parser.parse_known_args()

# Initialize random seed to 0 so that results are reproducible
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


from torch.utils.data import DataLoader, Dataset

"""
Reference: 
    https://medium.com/noumena/how-does-dataloader-work-in-pytorch-8c363a8ee6c1
"""

class your_dataset_class(Dataset):
    def __init__(self, imageroot, class_list, train, transform):
        """
        Make use of the arguments from argparse
        
        Initialize your program-defined variables 
        e.g., image path lists for cat and dog classes 
        
        Could also maintain label_array
        0 -- cat
        1 -- dog
        
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
        self.cat = 0
        self.dog = 1
        self.transform = transform
        self.train = train
        
        """ Here, define the image paths using the glob function, and transform them into tensors"""
        if self.train:
            pth = "/Train/"
        else:
            pth = "/Val/"
        
        self.x_filenames = glob.glob(self.imageroot + pth + "**/*.jpg")
        self.x_data = [self.transform(Image.open(filename)) for filename in self.x_filenames]
        
            
        """ Here, define the image paths using the glob function, and label the images"""
        self.i = 0
        self.y_data = torch.zeros(len(self.x_filenames))
        
        for clist in self.class_list: # in the command call we specify a list of classes of interest for training, and this goes through each of that class
            
            for name in glob.iglob(self.imageroot + pth + str(clist) + "/*.jpg", recursive=True): # this goes over each of the images in the folder
                
                if clist == "Cat":
                    self.y_data[self.i] = 0
                else:
                    self.y_data[self.i] = 1
                
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
        
import torchvision.transforms as tvt
        
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

batch_size = 10
train_dataset = your_dataset_class(args.imagenet_root, args.class_list, True, transform)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = your_dataset_class(args.imagenet_root, args.class_list, False, transform)
val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


""" Sub Task 2: Training """
dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 100 # feel free to adjust this parameter

D_in, H1, H2, D_out = 3*64*64, 1000, 256, 2

w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
w2 = torch.randn(H1, H2, device=device, dtype=dtype)
w3 = torch.randn(H2, D_out, device=device, dtype=dtype)

learning_rate = 1e-8

loss_cmm_list = []
my_data_file = open('output.txt', 'w')

for t in range(epochs):
    batch_idx, (images, targets) = next(enumerate(train_data_loader))
    loss_cmm = 0
    # correct_count = 0
    
    for idx in range(batch_size):
        images = images.to(device)
        targets = targets.to(device)

        x = images[idx].view(-1, D_in) # 10 x 3*64*64
        x = x.to(dtype = dtype) # converts from double to float
        
        h1 = torch.mm(x, w1)
        
        h1_relu = h1.clamp(min=0)
        
        h2 = torch.mm(h1_relu, w2)
        
        h2_relu = h2.clamp(min=0)
        
        y_pred = torch.mm(h2_relu, w3)
        # y_max = y_pred.max()
        # y_min = y_pred.min()
        
        # y_pred[y_pred == y_max] = 1
        # y_pred[y_pred == y_min] = 0
               
        if targets[idx] == 1.0: # if dog
            y = [1, 0]
            y = torch.FloatTensor(y)
            y = y.to(device)
            
        else: # if cat
            y = [0, 1]
            y = torch.FloatTensor(y)
            y = y.to(device)

        y_error = y_pred - y
        
        grad_w3 = h2_relu.t().mm(2 * y_error)
        
        h2_error = 2.0 * y_error.mm(w3.t())
        
        h2_error[h2_error < 0] = 0
        
        grad_w2 = h1_relu.t().mm(2 * h2_error)
        
        h1_error = 2.0 * h2_error.mm(w2.t())
        
        h1_error[h1_error < 0] = 0
        
        grad_w1 = x.t().mm(2 * h1_error)
        
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
        
        
    #     # Compute and print loss
        # loss = (y_pred - y).pow(2).sum().item()
        
        # if loss == 0:
        #     correct_count += 1
        
        loss_cmm += (y_pred - y).pow(2).sum().item()
       
        
        # import matplotlib.pyplot as plt
        # tensor_image = tvt.ToPILImage()(images[idx])
        # plt.imshow(tensor_image)
        # plt.show()
        
    # # print loss per epoch
    print("Epoch %d: \t %0.4f"%(t, loss_cmm))

    with open('output.txt', 'a') as f:
        f.write("Epoch %d: \t %0.4f\n"%(t, loss_cmm))
    
    # print("Accuracy %d: \t %0.4f %%"%(t, correct_count/batch_size*100))
    loss_cmm_list.append(loss_cmm)
    
# store layer weights in pickle file format
torch.save({"w1": w1, "w2":w2, "w3":w3}, "./wts.pkl")

import matplotlib.pyplot as plt
plt.plot(range(0,epochs),loss_cmm_list)
plt.show()

"""
Validation: Load the saved weights
"""
import torch

filename = ".\hw02_Sai_Mudumba\wts.pkl"
infile = open(filename,'rb')
new_dict = torch.load(infile)
w1 = new_dict["w1"]
w2 = new_dict["w2"]
w3 = new_dict["w3"]

batch_idx, (images, targets) = next(enumerate(val_data_loader))
loss_cmm = 0
correct_count = 0
batch_size = 10

for idx in range(batch_size):
    
    if targets[idx] == 1.0: # if dog
        y = [1, 0]
        y = torch.FloatTensor(y)
        y = y.to(device)
    else: # if cat
        y = [0, 1]
        y = torch.FloatTensor(y)
        y = y.to(device)
    
    images = images.to(device)
    targets = targets.to(device)

    x = images[idx].view(-1, D_in) # 10 x 3*64*64
    x = x.to(dtype = dtype) # converts from double to float
    
    h1 = torch.mm(x, w1)
    
    h1_relu = h1.clamp(min=0)
    
    h2 = torch.mm(h1_relu, w2)
    
    h2_relu = h2.clamp(min=0)
    
    y_pred = torch.mm(h2_relu, w3)
    
    loss_cmm += (y_pred - y).pow(2).sum().item()

    
    y_max = y_pred.max()
    y_min = y_pred.min()
    
    y_pred[y_pred == y_max] = 1
    y_pred[y_pred == y_min] = 0
    
    
    loss = (y_pred - y).pow(2).sum().item()

    if loss == 0:
        correct_count += 1    
    

print("\nVal Loss:\t %0.4f"%(loss_cmm))
print("Val Accuracy: \t %0.4f %%"%(correct_count/batch_size*100))
    

with open('output.txt', 'a') as f:
    f.write("\nVal Loss:\t %0.4f"%(loss_cmm))
    f.write("\nVal Accuracy: \t %0.4f %%"%(correct_count/batch_size*100))
    
    
    
"""
python hw02_imagenet_task2.py --imagenet_root C:/Users/"Sai Mudumba"/Documents/PS2 --class_list "Cat" "Dog"
"""