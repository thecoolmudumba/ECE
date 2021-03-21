# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:31:45 2021

@author: Sai Mudumba
"""

import torchvision
import torch.utils.data
import glob
import os
import numpy
import PIL
import argparse
import requests
import logging
import json

    
# use try and except

# Specifying import arguments to call these user specified input arguments from command line
parser = argparse.ArgumentParser(description='HW02 Task1')
parser.add_argument('--subclass_list', nargs='*', type=str, required=True)
parser.add_argument('--images_per_subclass', type=str, required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--main_class', type=str, required=True)
parser.add_argument('--imagenet_info_json', type=str, required=True)
args, args_other = parser.parse_known_args()


# Create the required directories
parent_directory = "C:/Users/Sai Mudumba/Documents/PS2/"
directory_Train = "Train"
train_path = os.path.join(parent_directory, directory_Train)
try:
    os.mkdir(train_path)
except OSError as error:
    None
    
parent_directory = "C:/Users/Sai Mudumba/Documents/PS2/"
directory_Val = "Val"
val_path = os.path.join(parent_directory, directory_Val)
try:
    os.mkdir(val_path)
except OSError as error:
    None
    
directory_Cat = "Cat"
Cat_path = os.path.join(train_path, directory_Cat)
try:
    os.mkdir(Cat_path)
except OSError as error:
    None
    
directory_Cat = "Cat"
Cat_path = os.path.join(val_path, directory_Cat)
try:
    os.mkdir(Cat_path)
except OSError as error:
    None
    
directory_Dog = "Dog"
Dog_path = os.path.join(train_path, directory_Dog)
try:
    os.mkdir(Dog_path)
except OSError as error:
    None
    
directory_Dog = "Dog"
Dog_path = os.path.join(val_path, directory_Dog)
try:
    os.mkdir(Dog_path)
except OSError as error:
    None



import requests
from PIL import Image
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
from os import path

def get_image(img_url, class_folder):
    '''
    Reference: https://github.com/johancc/ImageNetDownloader/blob/master/downloader.py
    '''
    
    if len(img_url) <= 1:
        # url is useless Do Something
        return None
        
    try:
        img_resp = requests.get(img_url, timeout = 1)
        
    except ConnectionError:
        # Handle this exception
        # print(f"Connection Error for URL {img_url}")
        return None 
    except ReadTimeout:
        # Handle this exception
        # print(f"Timeout Error for URL {img_url}")
        return None
    except TooManyRedirects:
        # Handle this exception
        # print(f"Redirect Error for URL {img_url}")
        return None
    except MissingSchema:
        # print(f"Schema Error for URL {img_url}")
        return None
    except InvalidURL:
        # print(f"URL Error for URL {img_url}")
        return None
    
    if not 'content-type' in img_resp.headers:
    # Missing content. Do Something
        # print(f"Content Error for URL {img_url}")
        return None
    if not 'image' in img_resp.headers['content-type']:
    # The URL doesn't have any image. Do Something
        # print(f"Image Error for URL {img_url}")
        return None
    if (len(img_resp.content) < 1000):
    # Ignore images < 1kb
        # print(f"Size Error for URL {img_url}")
        return None
        
    img_name = img_url.split('/')[-1]
    img_name = img_name.split("?")[0]

    if (len(img_name) <= 1):
    # Missing image name
        # print(f"Image Name Error for URL {img_url}")
        return None
    if not 'flickr' in img_url:
        # print(f"Some Other Error for URL {img_url}")
        return None
        
    img_file_path = os.path.join(class_folder, img_name)
    # print(img_file_path)
    
    if path.exists(img_file_path):
        # duplicate files 
        return None
    
    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)
        
    # Resize image to 64x64
    im = Image.open(img_file_path)
    
    if im.mode != "RGB":
        im = im.convert(mode="RGB")
        
    im_resized = im.resize((64,64), Image.BOX) # Resize the images here
    
    im_resized.save(img_file_path)
    return True

with open(args.imagenet_info_json) as f:
    class_idx = json.load(f)
    
# Call the user specified input arguments
len_subclass_list = len(args.subclass_list) # find the length of the subclass list

# the first for loop iterates over the specified subclass in the arguments
# the second for loop iterates over the identifiers from the JSON file to map the subclass name to the number
for i in range(0,len_subclass_list):
    subclass = args.subclass_list[i]
    for x, y in class_idx.items():
        if y.get("class_name") == str(subclass):
            identifier = x
            print("Yes")
            print(identifier)
            break   
    the_list_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + str(identifier)
    resp = requests.get(the_list_url)
    urls = [url.decode('utf-8') for url in resp.content.splitlines()]
    
    num_imgs = args.images_per_subclass # get the number of images
    
    count = 0 # start the count
    idx = 0 # start the index count; note that index could be more than images per subclass
    while count < int(num_imgs):
        ret = get_image(urls[idx],args.data_root)
        if ret == None:
            # print("Returned None")
            idx += 1
        else:
            print(f'Image {count} saved')
            count += 1
            idx += 1
# python hw02_ImageNet_Scrapper.py --subclass_list "Siamese cat" "Persian cat" "Burmese cat" --main_class ’cat’ --data_root C:/Users/"Sai Mudumba"/Documents/PS2/Train/Cat/ --imagenet_info_json C:/Users/"Sai Mudumba"/Documents/PS2/imagenet_class_info.json --images_per_subclass 200
# python hw02_ImageNet_Scrapper.py --subclass_list "hunting dog" "sporting dog" "shepherd dog" --main_class ’dog’ --data_root C:/Users/"Sai Mudumba"/Documents/PS2/Train/Dog/ --imagenet_info_json C:/Users/"Sai Mudumba"/Documents/PS2/imagenet_class_info.json --images_per_subclass 200

