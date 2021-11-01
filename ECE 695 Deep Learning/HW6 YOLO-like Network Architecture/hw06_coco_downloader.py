# -*- coding: utf-8 -*-
"""
Created on April 10, 20:02:19 2021

@author: Sai Mudumba
"""

"""
Import Libraries
1. argparse
2. os
3. requests
4. pycocotools.coco and COCO
5. numpy
6. skimage.io
7. matplotlib.pyplot
8. pylab
9. PIL and Image
"""
import argparse
import os
import requests
from pycocotools.coco import COCO
import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import cv2
import random
import torch
import pickle
import gzip
from matplotlib.pyplot import imshow

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

"""
Create the required directories of the following structure:
1. Train
    1.1. Image with A1, B1, C1 objects
    1.2. Image with A2, B2, C2 objects
    1.3. Image with A3, B3, C3 objects
    1.4. Image with A4, B4, C4 objects
        ...
    ...

2. Val
    2.1. Image with A1, B1, C1 objects
    2.2. Image with A2, B2, C2 objects
    2.3. Image with A3, B3, C3 objects
    2.4. Image with A4, B4, C4 objects
        ...
    ...

"""
def CreateDirectory(parent_folder, folder_name):
    """
    CreateDirectory function takes the parent_folder and the intended folder name and creates a folder
    """
    parent_directory = parent_folder
    child_directory = folder_name
    creater_path = os.path.join(parent_directory, child_directory)
    try:
        os.mkdir(creater_path)
    except OSError as error:
        None
    return creater_path

# This code creates folders 1 and 2, as specified in the structure above
COCO_Folder_Path = CreateDirectory("C:/Users/Sai Mudumba/Documents/PS06/", "COCO")
Train_Folder_Path = CreateDirectory(COCO_Folder_Path, "Train")
Val_Folder_Path = CreateDirectory(COCO_Folder_Path, "Val")

"""
python hw05_coco_downloader.py --root_path C:/Users/”Sai Mudumba”/Documents/PS05_v1/COCO/Train/ --coco_json_path C:/Users/”Sai Mudumba”/Documents/PS05_v1/COCO2017/annotations/instances_train2017.json --class_list “refrigerator” “airplane” “giraffe” “cat” “elephant” “dog” “train” “horse” “boat” “truck” --images_per_class 1
python hw05_coco_downloader.py --root_path C:/Users/”Sai Mudumba”/Documents/PS05_v1/COCO/Val/ --coco_json_path C:/Users/”Sai Mudumba”/Documents/PS05_v1/COCO2017/annotations/instances_val2017.json --class_list “refrigerator” “airplane” “giraffe” “cat” “elephant” “dog” “train” “horse” “boat” “truck” --images_per_class 50

"""

parser = argparse.ArgumentParser(description="HW06 COCO downloader")
parser.add_argument("--root_path", required=True, type=str)
parser.add_argument("--coco_json_path", required=True, type=str)
parser.add_argument("--class_list", required=True, nargs="*", type=str)
parser.add_argument("--images_per_class", required=True, type=int)
args, args_other = parser.parse_known_args()

class_list = args.class_list#['refrigerator', 'airplane', 'giraffe', 'cat', 'elephant', 'dog', 'train', 'horse', 'boat', 'truck']# args.class_list
# print(class_list)

# This code creates 1.1, 1.2. 1.3. ... and same for 2.1, 2.2, 2.3. ... as specified in the folder structure above
# Class_Paths_Train_List = [] # paths of all the classes for training are stored in a list
# Class_Paths_Val_List = [] # paths of all the classes for validation are stored in a list
# for i in range(len(class_list)): # i is the index of the class list, going from 0 to length of the list
#     Class_Paths_Train_List.append(CreateDirectory(Train_Folder_Path, class_list[i]))
#     Class_Paths_Val_List.append(CreateDirectory(Val_Folder_Path, class_list[i]))

        
"""
Now, need to fetch the images from COCO image dataset and save them in each of the classes created above
"""
annFile = args.coco_json_path

# initialize COCO api for instance annotations
coco=COCO(annFile)
dataset = []
bb_saved = {}
coco_labels_inverse = {}

# display COCO categories and supercategories
catIds = coco.getCatIds(catNms = class_list)
cats = coco.loadCats(catIds)
cats.sort(key=lambda x:x['id'])
# print(f"Category IDs: {catIds}")    
# print(f"Categories: {cats}")
# the above print(cats) prints a list of dictionaries in this form:
# [{'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]

for idx, in_class in enumerate(class_list):  # e.g., idx, in_class = 0, airplane
    for c in cats: # e.g., c = {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'} --> a dictionary
        if c['name'] == in_class:
            coco_labels_inverse[c['id']] = idx  # c['id'] = 89, so coco_labels_inverse['89'] = 0
# with the above code, one can expect a mapping of class id to the index e.g., {89: 0, 60: 1, 12: 2}. The following print statement should confirm that:
# print(f"Assigned Label for Each Category Id: {coco_labels_inverse}")

imgIds = coco.getImgIds(catIds = catIds)
# print(f"Image ID for each class{imgIds}") # don't print this because it's really long.

img = coco.loadImgs(imgIds[0:args.images_per_class])
# idx = np.random.randint(0,len(imgIds))
# img = coco.loadImgs(imgIds[idx])[0]

for im in img:
    I = skimage.io.imread(im['coco_url'])
    if len(I.shape) == 2:
        I = skimage.color.gray2rgb(I)

    """
    annotation id gives us the following:
    segmentation: a list of polygon vertices around the object (x, y, pixel location)
    area: area measured in pixels
    image_id: integer ID for COCO images
    bbox: bounding box ------> this is what we want
    id: annotation ID
    category_id: COCO Category ID
    iscrowd: is the segmentation for a single object or for a cluster of objects
    """
    annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd = False )
    anns = coco.loadAnns(annIds)
    # print(annIds)
    # print(anns)

    

    # if there are multiple bounding boxes in an image, this finds the one bounding box with max area
    bboxes = []
    area_bb = []
    for ann in anns:
        [x,y,w,h] = ann['bbox']
        area_bb.append(w*h)
        
    max_val = max(area_bb)  # maximum area is found here
    max_bb = area_bb.index(max_val) # the max area bounding box index is found
    
    area_bb_new = set(area_bb)
    area_bb_new.remove(max(area_bb))
    second_max_val = max(area_bb_new)
    second_max_bb = area_bb.index(second_max_val)     
    
    area_bb_new.remove(max(area_bb_new))
    third_max_val = max(area_bb_new)
    third_max_bb = area_bb.index(third_max_val)
    
    # 1st object
    [x,y,w,h] = anns[max_bb]['bbox']
    label = coco_labels_inverse[anns[max_bb]['category_id']]
    bboxes.append([x,y,w,h,label])
    
    # 2nd object
    [x,y,w,h] = anns[second_max_bb]['bbox']
    label = coco_labels_inverse[anns[second_max_bb]['category_id']]
    bboxes.append([x,y,w,h,label])
    
    # 3nd object
    [x,y,w,h] = anns[third_max_bb]['bbox']
    label = coco_labels_inverse[anns[third_max_bb]['category_id']]
    bboxes.append([x,y,w,h,label])
    
    # for ann in anns[0:2]:
    #     [x,y,w,h] = ann['bbox']
    #     label = coco_labels_inverse[ann['category_id']] # get the label = 0, 1, 2, 3, 4, 5... that we assigned
    #     [x,y,w,h].append(label)
    #     bboxes.append([x, y, w, h, label])
        # print(label)
        # print(area_bb)
    
    # bboxes = [ [x1, y1, w1, h1], [x2, y2, w2, h2], [x3, y3, w3, h3], ...  ]
    
    # max_val = max(area_bb)  # maximum area is found here
    # max_bb = area_bb.index(max_val) # the max area bounding box index is found
    # # print(max_bb)
    # [x,y,w,h] = anns[max_bb]['bbox'] # x, y, w, h are assigned to the max bb
    # print([x,y,x+w,x+h])

    # save the bounding box to a dictionary
    # bb_saved[int(ann['id'])] = [x,y,w,h]
    # bb_saved[im['file_name']] = [x,y,w,h]
    
    
    # label = coco_labels_inverse[ann['category_id']] # get the label = 0, 1, 2, 3, 4, 5... that we assigned
    # print(label)
    # print the image and do not show it yet, because we will also plot a bounding box on it
    
    # fig, ax = plt.subplots(1,1)
    # image = np.uint8(I)
    # for t in range(len(bboxes)):
    #     # plot the bounding box around the image
    #     image = cv2.rectangle(image, (int(bboxes[t][0]), int(bboxes[t][1])), (int(bboxes[t][0] + bboxes[t][2]), int(bboxes[t][1] + bboxes[t][3])), (36 ,255 ,12), 2)
    #     # put a label next to the bounding box
    #     image = cv2.putText(image, class_list[bboxes[t][4]], (int(bboxes[t][0]),int(bboxes[t][1]-10)), cv2.FONT_HERSHEY_SIMPLEX ,0.8, (36 ,255 ,12), 2)
    # ax.imshow(image) 
    # ax.set_axis_off()
    # plt.axis('tight')
    # plt.show()

    # e.g., save the image to the root_path: Train directory; class_list: airplane; image file name
    try:
        img_data = requests.get(im['coco_url'],timeout=50).content
    except ConnectionError:
        print("Had a connection error")
    with open(args.root_path + '/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)
    
    # print(f"Image {ann['id']} Saved: {im['file_name']}")
    
    sz = 64*2
    imgg = Image.open(args.root_path + '/' + im['file_name'])
    img_width, img_height = imgg.size
    
    bb_saved[im['file_name']] = [[bboxes[t][0]*sz/img_width,bboxes[t][1]*sz/img_height,bboxes[t][2]*sz/img_width,bboxes[t][3]*sz/img_height, bboxes[t][4]] for t in range(len(bboxes))]
    if imgg.mode != "RGB":
        imgg = imgg.convert(mode="RGB")
    im_resized = imgg.resize((sz,sz), Image.BOX) # Resize the images here
    im_resized.save(args.root_path + '/' + im['file_name'])
    
    # imgg = Image.open(args.root_path + '/' + im['file_name'])
    # fig, ax = plt.subplots(1,1)
    # imgg = np.asarray(imgg)
    
    # for t in range(len(bboxes)):   
    #     imgg = cv2.rectangle(imgg, (int(bboxes[t][0]*sz/img_width), int(bboxes[t][1]*sz/img_height)), (int(bboxes[t][0]*sz/img_width + bboxes[t][2]*sz/img_width), int(bboxes[t][1]*sz/img_height + bboxes[t][3]*sz/img_height)), (36 ,255 ,12), 1)
    #     # put a label next to the bounding box
    #     imgg = cv2.putText(imgg, class_list[bboxes[t][4]], (int(bboxes[t][0]*sz/img_width),int(bboxes[t][1]*sz/img_height-5)), cv2.FONT_HERSHEY_SIMPLEX ,0.3, (36 ,255 ,12), 1)
    # ax.imshow(imgg) 
    # ax.set_axis_off()
    # plt.axis('tight')
    # plt.show()
    
    """
    Extracting the Pixels from the Images        
    """
    imgg = Image.open(args.root_path + '/' + im['file_name'])
    data = list(imgg.getdata()) # it is a list of tuples containing RGB e.g., [(0,0,0), (0,0,0), ...]
    R = [pixel[0] for pixel in data]
    G = [pixel[1] for pixel in data]
    B = [pixel[2] for pixel in data]
    
    ## Find bounding rectangle
    # non_zero_pixels = []
    # for k,pixel in enumerate(data):
    #     x = k % 32
    #     y = k // 32
    #     if any( pixel[p] is not 0 for p in range(3) ):
    #         non_zero_pixels.append((x,y))
    # min_x = min( [pixel[0] for pixel in non_zero_pixels] )
    # max_x = max( [pixel[0] for pixel in non_zero_pixels] )
    # min_y = min( [pixel[1] for pixel in non_zero_pixels] )
    # max_y = max( [pixel[1] for pixel in non_zero_pixels] )
    
    # dataset.append([R, G, B, [min_x,min_y,max_x,max_y],label])
    dataset.append([R, G, B, [x*sz/img_width, y*sz/img_height, (x*sz/img_width)+(w*sz/img_width),(y*sz/img_height)+(h*sz/img_height)],label])
    # print(img_width, img_height, sz)
    # print([x*sz/img_width, y*sz/img_height, (x*sz/img_width)+(w*sz/img_width),(y*sz/img_height)+(h*sz/img_height)])
    # print([min_x,min_y,max_x,max_y])
# Save the bb_saved as a pickle file
# print(coco_labels_inverse)
        
# torch.save(bb_saved, "./wts.pkl")
# torch.save(dataset, "./dataset.pkl")
# torch.save(coco_labels_inverse, "./coco_labels_inverse.pkl")
# serialized = pickle.dumps([dataset, coco_labels_inverse])
# f = gzip.open("./dataset.gz", 'wb')
# f.write(serialized)
print(bb_saved)
torch.save(bb_saved, "./bboxes_train.p")
torch.save(bb_saved, "./dataset_val.pkl")
torch.save(coco_labels_inverse, "./coco_labels_inverse_val.pkl")
serialized = pickle.dumps(bb_saved)
f = gzip.open("./dataset_val.gz", 'wb')
f.write(serialized)

        # imgg = Image.open(args.root_path + class_list[i] + '/' + im['file_name'])
        # if imgg.mode != "RGB":
        #     imgg = imgg.convert(mode="RGB")
            
        # # im_resized = imgg.resize((64,64), Image.BOX) # Resize the images here
        
        # im_resized.save(args.root_path + class_list[i] + '/' + im['file_name'])
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# # get all images containing given categories, select one at random
# for i in range(0,len(class_list)):
#      catIds = coco.getCatIds(catNms=class_list[i]);
#      imgIds = coco.getImgIds(catIds=catIds);
#      img = coco.loadImgs(imgIds[0:args.images_per_class])

#     for im in img:
#         img_data = requests.get(im['coco_url']).content
#         with open(args.root_path + args.class_list[i] + '/' + im['file_name'], 'wb') as handler:
#             handler.write(img_data)
#         # Resize image to 64x64
#         imgg = Image.open(args.root_path + args.class_list[i] + '/' + im['file_name'])
        
#         if imgg.mode != "RGB":
#             imgg = imgg.convert(mode="RGB")
            
#         im_resized = imgg.resize((64,64), Image.BOX) # Resize the images here
        
#         im_resized.save(args.root_path + args.class_list[i] + '/' + im['file_name'])
