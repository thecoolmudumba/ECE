# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:02:19 2021

@author: Sai Mudumba
"""
import argparse
import os
import requests
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from PIL import Image

# Create the required directories
parent_directory = "C:/Users/Sai Mudumba/Documents/PS04/COCO/"
directory_Train = "Train"
train_path = os.path.join(parent_directory, directory_Train)
try:
    os.mkdir(train_path)
except OSError as error:
    None
    
parent_directory = "C:/Users/Sai Mudumba/Documents/PS04/COCO/"
directory_Val = "Val"
val_path = os.path.join(parent_directory, directory_Val)
try:
    os.mkdir(val_path)
except OSError as error:
    None

parser = argparse.ArgumentParser(description="HW04 COCO downloader")
parser.add_argument("--root_path", required=True, type=str)
parser.add_argument("--coco_json_path", required=True, type=str)
parser.add_argument("--class_list", required=True, nargs="*", type=str)
parser.add_argument("--images_per_class", required=True, type=int)
args, args_other = parser.parse_known_args()

for i in range(len(args.class_list)):
    directory_ten_classes = args.class_list[i]
    path_class = os.path.join(train_path, directory_ten_classes)
    try:
        os.mkdir(path_class)
    except OSError as error:
        None
    
for i in range(len(args.class_list)):
    directory_ten_classes = args.class_list[i]
    path_class = os.path.join(val_path, directory_ten_classes)
    try:
        os.mkdir(path_class)
    except OSError as error:
        None
        
        
# dataDir='..'
# dataType='2017'
annFile = args.coco_json_path #'{}COCO2017/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
for i in range(0,len(args.class_list)):
    catIds = coco.getCatIds(catNms=args.class_list[i]);
    imgIds = coco.getImgIds(catIds=catIds);
    img = coco.loadImgs(imgIds[0:args.images_per_class])

    for im in img:
        img_data = requests.get(im['coco_url']).content
        with open(args.root_path + args.class_list[i] + '/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)
        # Resize image to 64x64
        imgg = Image.open(args.root_path + args.class_list[i] + '/' + im['file_name'])
        
        if imgg.mode != "RGB":
            imgg = imgg.convert(mode="RGB")
            
        im_resized = imgg.resize((64,64), Image.BOX) # Resize the images here
        
        im_resized.save(args.root_path + args.class_list[i] + '/' + im['file_name'])
