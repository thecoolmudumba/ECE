# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:45:50 2021

@author: Sai Mudumba
"""
"""
Reference: https://engineering.purdue.edu/kak/distRPG/RegionProposalGenerator-2.0.1.html

Much of the code template came from the reference
"""
import argparse
import numpy as np
import torch
import os, sys
import torchvision.transforms as tvt
import glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gzip
import pickle
import copy
import torch.optim as optim
import torchvision  
import random
import time 
torch.set_printoptions(threshold=sys.maxsize)
import logging
import cv2
import skimage
import skimage.io as io


dataroot_train = "./COCO/"
dataroot_test  = "./COCO/"
image_size = [128,128]
yolo_interval = 20
path_saved_yolo_model = "./saved_yolo_model"
momentum = 0.9
learning_rate = 1e-3
epochs = 10
batch_size = 4
classes = ['airplane','truck','person']
use_gpu = True
device = torch.device("cuda:0")

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

class PurdueDrEvalMultiDataset(torch.utils.data.Dataset):        
    """
    This is the dataset to use if you are experimenting with multi-instance object
    detection.  As with the previous dataset, it contains three kinds of objects
    in its images: Dr. Eval, and two "objects" in his neighborhood: a house and a
    watertower.  Each 128x128 image in the dataset contains up to 5 instances of
    these objects. The instances are randomly scaled and colored and exact number
    of instances in each image is also chosen randomly. Subsequently, background
    clutter is added to the images --- these are again randomly chosen
    shapes. The number of clutter objects is also chosen randomly but cannot
    exceed 10.  In addition to the structured clutter, I add 20% Gaussian noise
    to each image.  Examples of these images are shown in Week 9 lecture material
    in Purdue's Deep Learning class.

    On account of the much richer structure of the image annotations, this
    dataset is organized very differently from the previous one:


                                      dataroot
                                         |
                                         |
                             ___________________________
                            |                           |
                            |                           |
                       annotations.p                  images


    Since each image is allowed to contain instances of the three different types
    of "meaningful" objects, it is not possible to organize the images on the
    basis of what they contain.

    As for the annotations, the annotation for each 128x128 image is a dictionary
    that contains information related to all the object instances in the image. Here
    is an example of the annotation for an image that has three instances in it:

        annotation:  {'filename': None, 
                      'num_objects': 3, 
                      'bboxes': {0: (67, 72, 83, 118), 
                                 1: (65, 2, 93, 26), 
                                 2: (16, 68, 53, 122), 
                                 3: None, 
                                 4: None}, 
                      'bbox_labels': {0: 'Dr_Eval', 
                                      1: 'house', 
                                      2: 'watertower', 
                                      3: None, 
                                      4: None}, 
                      'seg_masks': {0: <PIL.Image.Image image mode=1 size=128x128 at 0x7F5A06C838E0>, 
                                    1: <PIL.Image.Image image mode=1 size=128x128 at 0x7F5A06C837F0>, 
                                    2: <PIL.Image.Image image mode=1 size=128x128 at 0x7F5A06C838B0>, 
                                    3: None, 
                                    4: None}
                     }

    The annotations for the individual images are stored in a global Python
    dictionary called 'all_annotations' whose keys consist of the pathnames to
    the individual image files and the values the annotations dict for the
    corresponding images.  The filename shown above in the keystroke diagram,
    'annotations.p' is what you get by calling 'pickle.dump()' on the
    'all_annotations' dictionary.
    """

    def __init__(self, train_or_test, dataroot_train=None, dataroot_test=None, transform=None):
        super(PurdueDrEvalMultiDataset, self).__init__()
        self.train_or_test = train_or_test
        self.dataroot_train = dataroot_train
        self.dataroot_test  = dataroot_test
        self.database_train = {}
        self.database_test = {}
        self.dataset_size_train = None
        self.dataset_size_test = None
        if train_or_test == 'train':
            self.training_dataset = self.index_dataset()
        if train_or_test == 'test':
            self.testing_dataset = self.index_dataset()
        self.class_labels = None

    def index_dataset(self):
        if self.train_or_test == 'train':
            dataroot = self.dataroot_train
        elif self.train_or_test == 'test': 
            dataroot = self.dataroot_test
        if self.train_or_test == 'train' and dataroot == self.dataroot_train:
            if os.path.exists("torch_saved_Purdue_Dr_Eval_multi_dataset_train_10000.pt"):
                print("\nLoading training data from torch saved file")
                self.database_train = torch.load("torch_saved_Purdue_Dr_Eval_multi_dataset_train_10000.pt")
                self.dataset_size_train =  len(self.database_train)
            else: 
                print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                      """the dataset for this script. First time loading could take\n"""
                      """up to 3 minutes.  Any subsequent attempts will only take\n"""
                      """a few seconds.\n\n\n""")
                if os.path.exists(dataroot):      
                    root_dir = "C:/Users/Sai Mudumba/Documents/PS06/"
                    f = gzip.open(root_dir + "dataset_val.gz", 'rb')
                    dataset = f.read()
                    all_annotations = pickle.loads(dataset)# open( dataroot + 'bboxes_train.p', 'rb') )
                    # print(all_annotations)
                    all_image_paths = sorted(glob.glob(dataroot + "Train/*"))
                    all_image_names = [os.path.split(filename)[1] for filename in all_image_paths]
                    for idx,image_name in enumerate(all_image_names):        
                        # print(image_name)
                        annotation = all_annotations[image_name]
                        # print(annotation)
                        image_path = dataroot + "Train/" + image_name
                        self.database_train[idx] = [image_path, annotation]
                    all_training_images = list(self.database_train.values())
                    # print(all_training_images)
                    random.shuffle(all_training_images)
                    self.database_train = {i : all_training_images[i] for i in range(len(all_training_images))}
                    torch.save(self.database_train, "torch_saved_Purdue_Dr_Eval_multi_dataset_train_10000.pt")
                    self.dataset_size_train = len(all_training_images)
        elif self.train_or_test == 'test' and dataroot == self.dataroot_test:
            if os.path.exists(dataroot):      
                root_dir = "C:/Users/Sai Mudumba/Documents/PS06/"
                f = gzip.open(root_dir + "dataset_val.gz", 'rb')
                dataset = f.read()
                all_annotations = pickle.loads(dataset)    
                all_image_paths = sorted(glob.glob(dataroot + "Val/*"))
                all_image_names = [os.path.split(filename)[1] for filename in all_image_paths]                    
                for idx,image_name in enumerate(all_image_names):        
                    annotation = all_annotations[image_name]
                    image_path = dataroot + "Val/" + image_name
                    self.database_test[idx] =  [image_path, annotation]
                all_testing_images = list(self.database_test.values())
                random.shuffle(all_testing_images)
                self.database_test = {i : all_testing_images[i] for i in range(len(all_testing_images))}
                self.dataset_size_test = len(all_testing_images)

    def __len__(self):
        if self.train_or_test == 'train':
            return self.dataset_size_train
        elif self.train_or_test == 'test':
            return self.dataset_size_test

    def __getitem__(self, idx):
        if self.train_or_test == 'train':       
            image_path, annotation = self.database_train[idx]
        elif self.train_or_test == 'test':
            image_path, annotation = self.database_test[idx]
        im = Image.open(image_path)
        # im.show()
        im_tensor = tvt.ToTensor()(im)
        bbox_tensor     = torch.zeros(len(annotation),4, dtype=torch.uint8)
        bbox_label_tensor    = torch.zeros(3, dtype=torch.uint8) + 13
        num_objects_in_image = len(annotation)
        #print(f'Number of objects: {num_objects_in_image}')
        obj_class_labels = classes
        self.obj_class_label_dict = {obj_class_labels[i] : i for i in range(len(obj_class_labels))}
        # print(self.obj_class_label_dict)
        for i in range(num_objects_in_image):
            # print(annotation)
            bbox     = annotation[i]
            label    = annotation[i][4]
            # print(bbox)
            #print(f'Label: {label}')
            bbox_label_tensor[i] = label
            # seg_mask_tensor[i] = torch.from_numpy(seg_mask_arr)
            bbox_tensor[i] = torch.LongTensor(bbox[0:4])      
        # print(f'Im Tensor: {im_tensor.shape}')
        # print(f'bbox_tensor: {bbox_tensor}')
        # print(f'bbox_label_tensor: {bbox_label_tensor}')
        # print(f'bbox_label_tensor: {[classes[p] for p in bbox_label_tensor]}')
        # print(f'num_objects_in_image: {num_objects_in_image}')
        # print(f'This is the image id: {idx}')
        return im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image

class YoloLikeDetector(nn.Module):             
    """
    The primary purpose of this class is to demonstrate multi-instance object 
    detection with YOLO-like logic.  Toward that end, the following method
    of this class:

        run_code_for_training_multi_instance_detection()

    shows code based on my understanding of how YOLO works for multi-instance
    object detection.  A key parameter of the logic I have presented is in the
    form the variable 'yolo_interval'.  The image gridding that is required is
    based on the value assigned to this variable.  The grid is represented by an
    SxS array of cells where S is the image width divided by yolo_interval. So if
    'yolo_interval=20', you will get a 6x6 grid of cells over the image. [In the
    current version 0 of the implementation, I have not bothered with the bottom
    8 rows and the right-most 8 columns of the image that get left out of the
    area covered by such a grid.]

    Perhaps the most important element of the YOLO logic is defining a set of
    Anchor Boxes for each cell in the SxS grid.  The anchor boxes are
    characterized by their aspect ratios.  By aspect ratio I mean the
    'height/width' characterization of the boxes.  My implementation provides for
    5 anchor boxes for each cell with the following aspect ratios: 1/5, 1/3, 1/1,
    3/1, 5/1.  

    At training time, each instance in the image is assigned to that cell whose
    central pixel is closest to the center of the bounding box for the instance.
    After the cell assignment, the instance is assigned to that anchor box whose
    aspect ratio comes closest to matching the aspect ratio of the instance.

    The assigning of an object instance to a <cell, anchor_box> pair is encoded
    in the form of a '5+C' element long vector where C is the number of classes
    for the object instances.  In our cases, C is 3 for the three classes
    'Dr_Eval', 'house' and 'watertower', there we end up with an 8-element vector
    encoding when we assign an instance to a <cell, anchor_box> pair.  The last C
    elements of the encoding vector can be thought as a one-hot representation of
    the class label for the instance.

    The first five elements of the vector encoding for each anchor box in a cell
    are set as follows: The first element is 1 if an object instance was actually
    assigned to that anchor box. The next two elements are the (x,y)
    displacements of the center of the actual bounding box for the instance
    vis-a-vis the center of the cell. These two displacements are expressed as a
    fraction of the width and the height of the cell.  The next two elements of
    the vector encoding are the actual height and the actual width of the true
    bounding box for the instance in question as a multiple of the cell
    dimension.

    The 8-element vectors for each of the anchor boxes for all the cells in the
    SxS grid are concatenated together to form a large vector for regression.
    Since S is 6 in the implementation shown and we have 5 anchor boxes for each
    cell of the grid, we end up with a 1440 element tensor representation for
    each training image.

    """
    def __init__(self):
        super(YoloLikeDetector, self).__init__()
        self.train_dataloader = None
        self.test_dataloader = None

    def show_sample_images_from_dataset(self, rpg):
        data = next(iter(self.train_dataloader))    
        real_batch = data[0]
        first_im = real_batch[0]
        self.rpg.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

    def set_dataloaders(self, train=False, test=False):
        if train:
            dataserver_train = PurdueDrEvalMultiDataset("train", dataroot_train=dataroot_train)
            
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train, 
                                                  batch_size, shuffle=True)
        if test:
            dataserver_test = PurdueDrEvalMultiDataset("test", dataroot_test=dataroot_test)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test, 
                                                 batch_size, shuffle=False)

    def check_dataloader(self, how_many_batches_to_show, train=False, test=False):
        if train:      
            dataloader = self.train_dataloader
        if test:
            dataloader = self.test_dataloader
        for idx, data in enumerate(dataloader): 
            if idx >= how_many_batches_to_show:  
                break
            im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
            print("\n\nNumber of objects in the batch images: ", num_objects_in_image)
            print("\n\nlabels for the objects found:")
            print(bbox_label_tensor)

            mask_shape = seg_mask_tensor.shape
            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)
            #  Let's now display the batch images:
            plt.figure(figsize=[15,4])
            plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.show()
            #  Let's now display the batch with JUST the masks:
            composite_mask_tensor = torch.zeros(im_tensor.shape[0], 1,128,128)
            for bdx in range(im_tensor.shape[0]):
                for i in range(num_objects_in_image[bdx]):
                     composite_mask_tensor[bdx] += seg_mask_tensor[bdx][i]
            plt.figure(figsize=[15,4])
            plt.imshow(np.transpose(torchvision.utils.make_grid(composite_mask_tensor, normalize=True,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.show()
            #  Let's now display the batch and masks in a side-by-side display:
            display_image_and_mask_tensor = torch.zeros(2*im_tensor.shape[0], 3,128,128)
            display_image_and_mask_tensor[:im_tensor.shape[0],:,:,:]  = im_tensor
            display_image_and_mask_tensor[im_tensor.shape[0]:,:,:,:]  = composite_mask_tensor
            plt.figure(figsize=[15,4])
            plt.imshow(np.transpose(torchvision.utils.make_grid(display_image_and_mask_tensor, normalize=False,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.show()
            #  Let's now display the batch with GT bboxes for the objects:
            im_with_bbox_tensor = torch.clone(im_tensor)
            for bdx in range(im_tensor.shape[0]):
                bboxes_for_image = bbox_tensor[bdx]
                for i in range(num_objects_in_image[bdx]):
                    ii = bbox_tensor[bdx][i][0].item()
                    ji = bbox_tensor[bdx][i][1].item()
                    ki = bbox_tensor[bdx][i][2].item()
                    li = bbox_tensor[bdx][i][3].item()
                    im_with_bbox_tensor[bdx,:,ji,ii:ki] = 255    
                    im_with_bbox_tensor[bdx,:,li,ii:ki] = 255                
                    im_with_bbox_tensor[bdx,:,ji:li,ii] = 255  
                    im_with_bbox_tensor[bdx,:,ji:li,ki] = 255  
            plt.figure(figsize=[15,4])
            plt.imshow(np.transpose(torchvision.utils.make_grid(im_with_bbox_tensor, normalize=False,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.show()
            #  Let's now display the batch with GT bboxes and the object labels
            im_with_bbox_tensor = torch.clone(im_tensor)
            for bdx in range(im_tensor.shape[0]):
                labels_for_image = bbox_label_tensor[bdx]
                bboxes_for_image = bbox_tensor[bdx]
                for i in range(num_objects_in_image[bdx]):
                    ii = bbox_tensor[bdx][i][0].item()
                    ji = bbox_tensor[bdx][i][1].item()
                    ki = bbox_tensor[bdx][i][2].item()
                    li = bbox_tensor[bdx][i][3].item()
                    im_with_bbox_tensor[bdx,:,ji,ii:ki] = 40    
                    im_with_bbox_tensor[bdx,:,li,ii:ki] = 40                
                    im_with_bbox_tensor[bdx,:,ji:li,ii] = 40  
                    im_with_bbox_tensor[bdx,:,ji:li,ki] = 40  
                    im_pil = tvt.ToPILImage()(im_with_bbox_tensor[bdx]).convert('RGBA')
                    text = Image.new('RGBA', im_pil.size, (255,255,255,0))
                    draw = ImageDraw.Draw(text)
                    horiz = ki-10 if ki>10 else ki
                    vert = li
                    label = self.rpg.class_labels[labels_for_image[i]]
                    label = "wtower" if label == "watertower" else label
                    label = "Dr Eval" if label == "Dr_Eval" else label
                    draw.text( (horiz,vert), label, fill=(255,255,255,200) )
                    im_pil = Image.alpha_composite(im_pil, text)
                    im_with_bbox_tensor[bdx] = tvt.ToTensor()(im_pil.convert('RGB'))

            plt.figure(figsize=[15,4])
            plt.imshow(np.transpose(torchvision.utils.make_grid(im_with_bbox_tensor, normalize=False,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.show()
            logger.setLevel(old_level)


    class SkipBlock(nn.Module):
        """
        This is a building-block class that I have used in several networks
        """
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(YoloLikeDetector.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer1 = nn.BatchNorm2d
            norm_layer2 = nn.BatchNorm2d
            self.bn1 = norm_layer1(out_ch)
            self.bn2 = norm_layer2(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        def forward(self, x):
            identity = x                                     
            out = self.convo1(x)                              
            out = self.bn1(out)                              
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.convo2(out)                              
                out = self.bn2(out)                              
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity                              
                else:
                    out[:,:self.in_ch,:,:] += identity
                    out[:,self.in_ch:,:,:] += identity
            return out

    class NetForYolo(nn.Module):
        """
        The YOLO approach to multi-instance detection is based entirely on regression.  As
        was mentioned earlier in the comment block associated with the enclosing
        class, each image is represented by a 1440 element tensor that consists
        of 8-element encodings for each anchor box for every cell in the SxS
        gridding of an image.  The network I show below is a modification of the
        network class LOADnet presented earlier for the case that all we want to
        do is regression.
        """ 
        def __init__(self, skip_connections=True, depth=8):
            super(YoloLikeDetector.NetForYolo, self).__init__()
            if depth not in [8,10,12,14,16]:
                sys.exit("This network has only been tested for 'depth' values 8, 10, 12, 14, and 16")
            self.depth = depth // 2
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.bn1  = nn.BatchNorm2d(64)
            self.bn2  = nn.BatchNorm2d(128)
            self.bn3  = nn.BatchNorm2d(256)
            self.skip64_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip64_arr.append(YoloLikeDetector.SkipBlock(64, 64,
                                                                                skip_connections=skip_connections))
            self.skip64ds = YoloLikeDetector.SkipBlock(64,64,downsample=True, 
                                                                                 skip_connections=skip_connections)
            self.skip64to128 = YoloLikeDetector.SkipBlock(64, 128, 
                                                                                skip_connections=skip_connections )
            self.skip128_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip128_arr.append(YoloLikeDetector.SkipBlock(128,128,
                                                                                skip_connections=skip_connections))
            self.skip128ds = YoloLikeDetector.SkipBlock(128,128,
                                                                downsample=True, skip_connections=skip_connections)
            self.skip128to256 = YoloLikeDetector.SkipBlock(128, 256, 
                                                                                skip_connections=skip_connections )
            self.skip256_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip256_arr.append(YoloLikeDetector.SkipBlock(256,256,
                                                                                skip_connections=skip_connections))
            self.skip256ds = YoloLikeDetector.SkipBlock(256,256,
                                                                downsample=True, skip_connections=skip_connections)
            self.fc_seqn = nn.Sequential(
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 1440)            )

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv1(x)))          
            x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv2(x)))       
            for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                x = skip64(x)                
            x = self.skip64ds(x)
            for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                x = skip64(x)                
            x = self.bn1(x)
            x = self.skip64to128(x)
            for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                x = skip128(x)                
            x = self.bn2(x)
            x = self.skip128ds(x)
            x = x.view(-1, 8192 )
            x = self.fc_seqn(x)
            return x
        
        
    def IoULossCalc(self, inputs, targets):        
        composite_loss = []
        for idx in range(3):
            union = intersection = 0.0
            for i in range(128):
                for j in range(128):
                    inp = inputs[idx,i,j]
                    tap = targets[idx,i,j]
                    if (inp == tap) and (inp==1):
                        intersection += 1
                        union += 1
                    elif (inp != tap) and ((inp==1) or (tap==1)):
                        union += 1
            if union == 0.0:
                raise Exception("something_wrong")
            batch_sample_iou = intersection / float(union)
            print(f'IOU Loss for Instance {idx}: {batch_sample_iou}')
            composite_loss.append(batch_sample_iou)
        total_iou_for_batch = sum(composite_loss) 
        return 1 - torch.tensor([total_iou_for_batch / batch_size])
        
        # union = intersection = 0.0
        # for i in range(image_size[0]):
        #     for j in range(image_size[1]):
        #         inp = inputs[i,j]
        #         tap = targets[i,j]
        #         if (inp == tap) and (inp==1):
        #             intersection += 1
        #             union += 1
        #         elif (inp != tap) and ((inp==1) or (tap==1)):
        #             union += 1
        # if union == 0.0:
        #     raise Exception("something_wrong")
        # batch_sample_iou = intersection / float(union)
        # return 1 - torch.tensor([batch_sample_iou / batch_size])

    def run_code_for_training_multi_instance_detection(self, net, display_images=True):        
        yolo_debug = False
        filename_for_out1 = "performance_numbers_" + str(epochs) + "label.txt"
        filename_for_out2 = "performance_numbers_" + str(epochs) + "regres.txt"
        FILE1 = open(filename_for_out1, 'w')
        FILE2 = open(filename_for_out2, 'w')
        net = copy.deepcopy(net)
        net = net.to(device)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        print("\n\nStarting training loop...\n\n")
        start_time = time.perf_counter()
        Loss_tally = []
        elapsed_time = 0.0
        num_yolo_cells = (image_size[0] // yolo_interval) * (image_size[1] // yolo_interval)
        num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
        #  The 8 in the following is the size of the yolo_vector for each
        #  anchor-box in a given cell.  The 8 elements are: [obj_present, bx, by,
        #  bh, bw, c1, c2, c3] where bx and by are the delta diffs between the centers
        #  of the yolo cell and the center of the object bounding box in terms of
        #  a unit for the cell width and cell height.  bh and bw are the height 
        #  and the width of object bounding box in terms of the cell height and width.
        yolo_tensor = torch.zeros( batch_size, num_yolo_cells, num_anchor_boxes, 8 )
        
        #print("I work?")
        class AnchorBox:
            """
            About the role of the 'adx' constructor parameter:  Recall that our goal is to use
            the annotations for each batch to fill up the 'yolo_tensor' that was defined above.
            For case of 5 anchor boxes per cell, this tensor has the following shape:

                     torch.zeros( self.rpg.batch_size, num_yolo_cells, 5, 8 )

            The index 'adx' shown below tells us which of the 5 dimensions on the third axis
            of the 'yolo_tensor' be RESERVED for an anchor box.  We will reserve the 
            coordinate 0 on the third axis for the "1/1" anchor boxes, the coordinate 1 for
            the "1/3" anchor boxes, and so on.  This coordinate choice is set by 'adx'. 
            """
            #               aspect_ratio top_left_corner  anchor_box height & width   anchor_box index
            def __init__(self,   AR,     tlc,             ab_height,   ab_width,      adx):     
                self.AR = AR
                self.tlc = tlc
                self.ab_height = ab_height
                self.ab_width = ab_width
                self.adx = adx
            def __str__(self):
                return "AnchorBox type (h/w): %s    tlc for yolo cell: %s    anchor-box height: %d     \
                   anchor-box width: %d   adx: %d" % (self.AR, str(self.tlc), self.ab_height, self.ab_width, self.adx)

        for epoch in range(epochs):  
            #print("I ran?")
            running_loss = 0.0
            for iter, data in enumerate(self.train_dataloader):
                #print("I initialized?")
                im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                im_tensor   = im_tensor.to(device)
                bbox_tensor = bbox_tensor.to(device)
                bbox_label_tensor = bbox_label_tensor.to(device)
                yolo_tensor = yolo_tensor.to(device)
                if yolo_debug:
                    logger = logging.getLogger()
                    old_level = logger.level
                    logger.setLevel(100)
                    plt.figure(figsize=[15,4])
                    plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
                    plt.show()
                    logger.setLevel(old_level)
                cell_height = yolo_interval
                cell_width = yolo_interval
                obj_centers = {ibx : {idx : None for idx in range(num_objects_in_image[ibx])} 
                                                                   for ibx in range(im_tensor.shape[0])}
                # print(f'Object centers: {obj_centers}')
                ##  2D indexing for access to the anchor boxes is implicit:
                ##  ALSO:  Aspect Ratio (AR) is the ratio of the height to the width of a bounding box
                ##         Therefore, in the name "anchor_boxes_1_3", AR is 1/3 when height/width = 1/3
                num_cells_image_width = image_size[0] // yolo_interval
                num_cells_image_height = image_size[1] // yolo_interval
                ## The pair in the second arg to the AnchorBox constructor are the pixel coordinates of the
                ## top-left corner of the yolo cell in question.

                anchor_boxes_1_1 = [[AnchorBox("1/1", (i*yolo_interval,j*yolo_interval), yolo_interval, yolo_interval,  0) 
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_1_3 = [[AnchorBox("1/3",(i*yolo_interval,j*yolo_interval), yolo_interval, 3*yolo_interval, 1) 
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_3_1 = [[AnchorBox("3/1",(i*yolo_interval,j*yolo_interval), 3*yolo_interval, yolo_interval, 2)
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_1_5 = [[AnchorBox("1/5",(i*yolo_interval,j*yolo_interval), yolo_interval, 5*yolo_interval, 3)
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_5_1 = [[AnchorBox("5/1",(i*yolo_interval,j*yolo_interval), 5*yolo_interval, yolo_interval, 4) 
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                
                #print("I worked")
                bbox_actual = []
                for ibx in range(im_tensor.shape[0]):
                    bbox_actual_ibx = []
                    for idx in range(3):
                        # print(f'Number of objects in image: {ibx} {idx} {num_objects_in_image[ibx]}')
                        ## Note that the bounding-box coordinates are in the (x,y) format, with x-positive going to
                        ## right and the y-positive going down. A bbox is specified by (x_min,y_min,x_max,y_max):
                        print(f'Batch {ibx}, Object No. {idx} {num_objects_in_image[ibx]}')
                        height_center_bb =  (bbox_tensor[ibx][idx][1].item() + bbox_tensor[ibx][idx][3].item()) // 2
                        width_center_bb =  (bbox_tensor[ibx][idx][0].item() + bbox_tensor[ibx][idx][2].item()) // 2
                        obj_bb_height = bbox_tensor[ibx][idx][3].item() -  bbox_tensor[ibx][idx][1].item()
                        obj_bb_width = bbox_tensor[ibx][idx][2].item() - bbox_tensor[ibx][idx][0].item()
                        

                        label = classes[bbox_label_tensor[ibx][idx].item()]
                        if (obj_bb_height < 4) or (obj_bb_width < 4):
                            continue
                        AR = float(obj_bb_height) / float(obj_bb_width)
                        if yolo_debug:
                            print("\n\n[Image ibx: %d  object: %d]   obj label: %s   obj_bb_height: %d    obj_bb_width: %d" % 
                                                                       (ibx, idx, label, obj_bb_height, obj_bb_width))
                        ## The following cell_row_idx and cell_col_idx are in the (i,j) format with i being the row
                        ## index for a pixel in the image and j being the column index.  The variables
                        ## cell_row_idx and cell_col_idx refer to index of the cell to be used in the yolo_interval
                        ## based gridding of the image.  The anchor boxes are centered at the centers of the grid cells:

                        cell_row_idx =  height_center_bb // yolo_interval          ## for the i coordinate
                        cell_col_idx = width_center_bb // yolo_interval            ## for the j coordinates
                        cell_row_idx = 5 if cell_row_idx > 5 else cell_row_idx
                        cell_col_idx = 5 if cell_col_idx > 5 else cell_col_idx

                        bbox_actual_ibx.append([cell_col_idx*20-obj_bb_width/2, cell_row_idx*20-obj_bb_height/2, obj_bb_width, obj_bb_height])


                        if AR <= 0.2:
                            anchbox = anchor_boxes_1_5[cell_row_idx][cell_col_idx]
                        elif AR <= 0.5:
                            anchbox = anchor_boxes_1_3[cell_row_idx][cell_col_idx]
                        elif AR <= 1.5:
                            anchbox = anchor_boxes_1_1[cell_row_idx][cell_col_idx]
                        elif AR <= 4:
                            anchbox = anchor_boxes_3_1[cell_row_idx][cell_col_idx]
                        elif AR > 4:
                            anchbox = anchor_boxes_5_1[cell_row_idx][cell_col_idx]

                        if yolo_debug:
                            print("iter=%d  cell_row_idx: " % iter, cell_row_idx)
                            print("iter=%d  cell_col_idx: " % iter, cell_col_idx)
                            print("iter=%d  Anchor box used for image: %d and object: %d: " % (iter, ibx,idx), anchbox)
                        # The bh and bw elements in the yolo vector for this object:
                        # bh and bw are measured relative to the size of the grid cell to which the object is
                        # assigned.  For example, bh is the height of the bounding-box divided by the actual
                        # heigh of the grid cell.
                        bh  =  float(obj_bb_height) / float(yolo_interval)
                        bw  =  float(obj_bb_width)  / float(yolo_interval)
                        # You have to be CAREFUL about object center calculation since bounding-box coordinates
                        # are in (x,y) format --- with x-positive going to the right and y-positive going down.
                        obj_center_x =  float(bbox_tensor[ibx][idx][2].item() +  bbox_tensor[ibx][idx][0].item()) / 2.0
                        obj_center_y =  float(bbox_tensor[ibx][idx][3].item() + bbox_tensor[ibx][idx][1].item()) / 2.0

                        # Now you need to switch back from (x,y) format to (i,j) format:
                        yolocell_center_i =  cell_row_idx*yolo_interval + float(yolo_interval) / 2.0
                        yolocell_center_j =  cell_col_idx*yolo_interval + float(yolo_interval) / 2.0
                        if yolo_debug:
                            print("object center located at: obj_center_x: %.3f    obj_center_y: %0.3f" % 
                                                                                      (obj_center_x, obj_center_y))
                            print("yolocell center located at: yolocell_center_j: %.3f   yolocell_center_i: %0.3f" % 
                                                                            (yolocell_center_j, yolocell_center_i))
                        del_x = float(obj_center_x - yolocell_center_j) / yolo_interval
                        del_y = float(obj_center_y - yolocell_center_i) / yolo_interval
                        yolo_vector = [1, del_x, del_y, bh, bw, 0, 0, 0]
                        yolo_vector[5 + bbox_label_tensor[ibx][idx].item()] = 1
                        ## Remember because the coordinate reversal between (x,y) and (i,j) formats, cell_row_idx
                        ## is the index along the horizontal dimension and cell_col_idx is along the vertical dimension.
                        yolo_cell_index =  cell_row_idx * num_cells_image_width  +  cell_col_idx
                        if yolo_debug:
                            print("iter=%d  yolo_vector: " % iter, yolo_vector)
                            print("iter=%d  yolo_cell_index: " % iter, yolo_cell_index)
                        yolo_tensor[ibx, yolo_cell_index, anchbox.adx] = torch.FloatTensor( yolo_vector )
                        # print(yolo_tensor)
                    bbox_actual.append(bbox_actual_ibx)
                print(f'Actual BBOX: {bbox_actual}')
                if yolo_debug:
                    logger = logging.getLogger()
                    old_level = logger.level
                    logger.setLevel(100)
                    plt.figure(figsize=[15,4])
                    plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                     padding=3, pad_value=255).cpu(), (1,2,0)))
                    plt.show()
                
                #print("I worked fine until here")
                yolo_tensor_flattened = yolo_tensor.view(im_tensor.shape[0], -1)
                optimizer.zero_grad()
                output = net(im_tensor)
                # print(f'Output: {output}')
                # print(f'GT: {yolo_tensor_flattened}')
                loss = criterion2(output, yolo_tensor_flattened)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if iter%5==4:    
                    if display_images:
                        print("\n\n\n")
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time 
                    avg_loss = running_loss / float(100)
                    print("[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean MSE loss: %7.4f" % 
                                                              (epoch+1,epochs, iter+1, elapsed_time, avg_loss))
                    Loss_tally.append(running_loss)
                    FILE1.write("%.3f\n" % avg_loss)
                    FILE1.flush()
                    running_loss = 0.0
                    if display_images:
                        predictions = output.view(4,36,5,8)
                        # print(predictions[0][0][0])
                        for ibx in range(predictions.shape[0]):                             # for each batch image
                            icx_2_best_anchor_box = {ic : None for ic in range(36)}
                            for icx in range(predictions.shape[1]):                         # for each yolo cell
                                cell_predi = predictions[ibx,icx]
                                # print(f'{icx}: {cell_predi.shape}')
                                prev_best = 0
                                for anchor_bdx in range(cell_predi.shape[0]):
                                    if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                        prev_best = anchor_bdx
                                best_anchor_box_icx = prev_best 
                                # print(best_anchor_box_icx)
                                # print(best_anchor_box_icx)
                                icx_2_best_anchor_box[icx] = best_anchor_box_icx
                            sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                       key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                            retained_cells = sorted_icx_to_box[:3]
                            # print(f'{icx_2_best_anchor_box}')
                            # print(f'{sorted_icx_to_box}')
                            # print(f'{retained_cells}')
                            objects_detected = []
                            bboxes_pred = []
                            cell_center_pred_idx = []
                            cell_center_pred = []
                            bbox_pred_center = []
                            
                            for icx in retained_cells:
                                cell_col_idx_pred = icx // 6 
                                cell_row_idx_pred = icx % 6
                                cell_center_pred_idx.append([cell_row_idx_pred, cell_col_idx_pred])
                                
                                cell_col_pred = cell_col_idx_pred * 20 + 10
                                cell_row_pred = cell_row_idx_pred * 20 + 10
                                cell_center_pred.append([cell_row_pred, cell_col_pred])
                                
                                pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                                # print(f'Prediction Vector for each bbox for image {ibx}: {pred_vec}')
                                
                                dx = pred_vec[1].item()
                                dy = pred_vec[2].item()
                                dw = pred_vec[3].item()
                                dh = pred_vec[4].item()
                                bboxes_pred.append([dx, dy, dw, dh])
                                
                                bbox_x = 20 * dx + cell_col_pred
                                bbox_y = 20 * dy + cell_row_pred
                                bbox_w = 20 * dw
                                bbox_h = 20 * dh
                                
                                bbox_pred_center.append([bbox_x - bbox_w/2, bbox_y - bbox_h/2, bbox_w, bbox_h])
                                
                                
                                class_labels_predi  = pred_vec[-3:]                        
                                
                                if torch.all(class_labels_predi < 0.2): 
                                    predicted_class_label = None
                                else:
                                    best_predicted_class_index = (class_labels_predi == 
                                                                      class_labels_predi.max()).nonzero().squeeze().item()
                                    predicted_class_label = classes[best_predicted_class_index]
                                    objects_detected.append(predicted_class_label)
#                                    print("[batch image=%d] most probable object instance detected: ", objects_detected[-1])
#                                    if yolo_debug:
                            print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                            print(bboxes_pred)
                            print(f'cell center index {cell_center_pred_idx}')
                            print(f'cell center {cell_center_pred}')
                            print(f'bbox predicted {bbox_pred_center}')
                            print(f'bbox actual {bbox_tensor}')
                            print(f'bbox actual {bbox_tensor[0]}')
                            print(f'bbox actual label {bbox_label_tensor}')
                    
                    # create masks
                    inputs = torch.zeros([128,128])
                    targets = torch.zeros([128,128])
                    
                    for i in range(128):
                        for j in range(128):
                            if (j >= bbox_tensor[0][0][0] and j <= bbox_tensor[0][0][0] + bbox_tensor[0][0][2]) and (i >= bbox_tensor[0][0][1] and i <= bbox_tensor[0][0][1] + bbox_tensor[0][0][3]):
                                inputs[i,j] = 1
                                print(inputs[i,j])
                            else:
                                inputs[i,j] = 0
                            
                            if (j >= bbox_pred_center[0][0] and j <= bbox_pred_center[0][0] + bbox_pred_center[0][2]) and (i >= bbox_pred_center[0][1] and i <= bbox_pred_center[0][1] + bbox_pred_center[0][3]):
                                targets[i,j] = 1
                            else:
                                targets[i,j] = 0
                    # print(inputs, targets)
                    IoULoss = self.IoULossCalc(inputs, targets)
                    print(f'{IoULoss} is the IOU loss')
                    
                    
                    if display_images:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[150,40])
                        
                        # image = np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True, padding=3, pad_value=255).cpu(), (1,2,0))
                        # image = skimage.color.gray2rgb(im_tensor[0].cpu())
                        # imgg = np.asarray(image)
                        # imgg = np.uint8(imgg)
                        im_tensor = im_tensor[0].cpu()
                        image = im_tensor.numpy()
                        image = image.transpose(1,2,0)
                        fig, ax = plt.subplots(1,1)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        # plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                        # padding=3, pad_value=255).cpu(), (1,2,0)))                        
                        # plt.imshow(np.transpose(im_tensor[0].cpu()))
                        for t in range(3):
                            # plot the bounding box around the image
                            image = cv2.rectangle(image, (int(bbox_pred_center[t][0]), int(bbox_pred_center[t][1])), (int(bbox_pred_center[t][0] + bbox_pred_center[t][2]), int(bbox_pred_center[t][1] + bbox_pred_center[t][3])), (255 ,0 ,0), 2)
                            image = cv2.rectangle(image, (int(bbox_tensor[0][t][0]), int(bbox_tensor[0][t][1])), (int(bbox_tensor[0][t][0] + bbox_tensor[0][t][2]), int(bbox_tensor[0][t][1] + bbox_tensor[0][t][3])), (0 ,0 ,255), 2)

                            if len(objects_detected) == 3:
                            # put a label next to the bounding box
                                image = cv2.putText(image, objects_detected[t], (int(bbox_pred_center[t][0]),int(bbox_pred_center[t][1]-5)), cv2.FONT_HERSHEY_SIMPLEX ,0.3, (255 , 0 ,0), 1)
                                image = cv2.putText(image, classes[bbox_label_tensor[0][t]], (int(bbox_tensor[0][t][0]),int(bbox_tensor[0][t][1]-5)), cv2.FONT_HERSHEY_SIMPLEX ,0.3, (0, 0, 255), 1)
                                
                        
                        plt.imshow(image)
                        logger.setLevel(old_level)

        print("\nFinished Training\n")
        print(Loss_tally)
        plt.figure(figsize=(10,5))
        plt.title("Loss vs. Iterations")
        plt.plot(Loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("training_loss.png")
        plt.show()
        torch.save(net.state_dict(), path_saved_yolo_model)
        return net
    
    def run_code_for_testing_multi_instance_detection(self, net, display_images=False):        
        net.load_state_dict(torch.load(path_saved_yolo_model))
        net = net.to(device)
        num_yolo_cells = (image_size[0] // yolo_interval) * (image_size[1] // yolo_interval)
        num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
        yolo_tensor = torch.zeros(batch_size, num_yolo_cells, num_anchor_boxes, 8 )
        
        iouLoss = []
        accuracy_list = []
        confusion_matrix = torch.zeros(len(classes), 
                                   len(classes))
        correct = 0
        total = 0
        
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)
        
        class AnchorBox:
            """
            About the role of the 'adx' constructor parameter:  Recall that our goal is to use
            the annotations for each batch to fill up the 'yolo_tensor' that was defined above.
            For case of 5 anchor boxes per cell, this tensor has the following shape:

                     torch.zeros( self.rpg.batch_size, num_yolo_cells, 5, 8 )

            The index 'adx' shown below tells us which of the 5 dimensions on the third axis
            of the 'yolo_tensor' be RESERVED for an anchor box.  We will reserve the 
            coordinate 0 on the third axis for the "1/1" anchor boxes, the coordinate 1 for
            the "1/3" anchor boxes, and so on.  This coordinate choice is set by 'adx'. 
            """
            #               aspect_ratio top_left_corner  anchor_box height & width   anchor_box index
            def __init__(self,   AR,     tlc,             ab_height,   ab_width,      adx):     
                self.AR = AR
                self.tlc = tlc
                self.ab_height = ab_height
                self.ab_width = ab_width
                self.adx = adx
            def __str__(self):
                return "AnchorBox type (h/w): %s    tlc for yolo cell: %s    anchor-box height: %d     \
                   anchor-box width: %d   adx: %d" % (self.AR, str(self.tlc), self.ab_height, self.ab_width, self.adx)
                   
                   
        
        for iter, data in enumerate(self.test_dataloader):
            im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
            if iter % 5 == 4:
                print("\n\n\n\nShowing output for test batch %d: " % (iter+1))
                im_tensor   = im_tensor.to(device)
                bbox_tensor = bbox_tensor.to(device)
                bbox_label_tensor = bbox_label_tensor.to(device)
                yolo_tensor = yolo_tensor.to(device)
                cell_height = yolo_interval
                cell_width = yolo_interval
                obj_centers = {ibx : {idx : None for idx in range(num_objects_in_image[ibx])} 
                                                                   for ibx in range(im_tensor.shape[0])}
                # print(f'Object centers: {obj_centers}')
                ##  2D indexing for access to the anchor boxes is implicit:
                ##  ALSO:  Aspect Ratio (AR) is the ratio of the height to the width of a bounding box
                ##         Therefore, in the name "anchor_boxes_1_3", AR is 1/3 when height/width = 1/3
                num_cells_image_width = image_size[0] // yolo_interval
                num_cells_image_height = image_size[1] // yolo_interval
                ## The pair in the second arg to the AnchorBox constructor are the pixel coordinates of the
                ## top-left corner of the yolo cell in question.

                anchor_boxes_1_1 = [[AnchorBox("1/1", (i*yolo_interval,j*yolo_interval), yolo_interval, yolo_interval,  0) 
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_1_3 = [[AnchorBox("1/3",(i*yolo_interval,j*yolo_interval), yolo_interval, 3*yolo_interval, 1) 
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_3_1 = [[AnchorBox("3/1",(i*yolo_interval,j*yolo_interval), 3*yolo_interval, yolo_interval, 2)
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_1_5 = [[AnchorBox("1/5",(i*yolo_interval,j*yolo_interval), yolo_interval, 5*yolo_interval, 3)
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                anchor_boxes_5_1 = [[AnchorBox("5/1",(i*yolo_interval,j*yolo_interval), 5*yolo_interval, yolo_interval, 4) 
                                                                    for i in range(0,num_cells_image_height)]
                                                                       for j in range(0,num_cells_image_width)]
                
                output = net(im_tensor)
                
                predictions = output.view(4,36,5,8)

                    

                bbox_actual = []
                for ibx in range(im_tensor.shape[0]):
                    bbox_actual_ibx = []
                    for idx in range(3):
                        # print(f'Number of objects in image: {ibx} {idx} {num_objects_in_image[ibx]}')
                        ## Note that the bounding-box coordinates are in the (x,y) format, with x-positive going to
                        ## right and the y-positive going down. A bbox is specified by (x_min,y_min,x_max,y_max):
                        # print(f'Batch {ibx}, Object No. {idx} {num_objects_in_image[ibx]}')
                        height_center_bb =  (bbox_tensor[ibx][idx][1].item() + bbox_tensor[ibx][idx][3].item()) // 2
                        width_center_bb =  (bbox_tensor[ibx][idx][0].item() + bbox_tensor[ibx][idx][2].item()) // 2
                        obj_bb_height = bbox_tensor[ibx][idx][3].item() -  bbox_tensor[ibx][idx][1].item()
                        obj_bb_width = bbox_tensor[ibx][idx][2].item() - bbox_tensor[ibx][idx][0].item()
                        

                        label = classes[bbox_label_tensor[ibx][idx].item()]
                        if (obj_bb_height < 4) or (obj_bb_width < 4):
                            continue
                        AR = float(obj_bb_height) / float(obj_bb_width)
                        # if yolo_debug:
                        #     print("\n\n[Image ibx: %d  object: %d]   obj label: %s   obj_bb_height: %d    obj_bb_width: %d" % 
                        #                                                (ibx, idx, label, obj_bb_height, obj_bb_width))
                        ## The following cell_row_idx and cell_col_idx are in the (i,j) format with i being the row
                        ## index for a pixel in the image and j being the column index.  The variables
                        ## cell_row_idx and cell_col_idx refer to index of the cell to be used in the yolo_interval
                        ## based gridding of the image.  The anchor boxes are centered at the centers of the grid cells:

                        cell_row_idx =  height_center_bb // yolo_interval          ## for the i coordinate
                        cell_col_idx = width_center_bb // yolo_interval            ## for the j coordinates
                        cell_row_idx = 5 if cell_row_idx > 5 else cell_row_idx
                        cell_col_idx = 5 if cell_col_idx > 5 else cell_col_idx

                        bbox_actual_ibx.append([cell_col_idx*20-obj_bb_width/2, cell_row_idx*20-obj_bb_height/2, obj_bb_width, obj_bb_height])


                        if AR <= 0.2:
                            anchbox = anchor_boxes_1_5[cell_row_idx][cell_col_idx]
                        elif AR <= 0.5:
                            anchbox = anchor_boxes_1_3[cell_row_idx][cell_col_idx]
                        elif AR <= 1.5:
                            anchbox = anchor_boxes_1_1[cell_row_idx][cell_col_idx]
                        elif AR <= 4:
                            anchbox = anchor_boxes_3_1[cell_row_idx][cell_col_idx]
                        elif AR > 4:
                            anchbox = anchor_boxes_5_1[cell_row_idx][cell_col_idx]

                        # if yolo_debug:
                        #     print("iter=%d  cell_row_idx: " % iter, cell_row_idx)
                        #     print("iter=%d  cell_col_idx: " % iter, cell_col_idx)
                        #     print("iter=%d  Anchor box used for image: %d and object: %d: " % (iter, ibx,idx), anchbox)
                        # The bh and bw elements in the yolo vector for this object:
                        # bh and bw are measured relative to the size of the grid cell to which the object is
                        # assigned.  For example, bh is the height of the bounding-box divided by the actual
                        # heigh of the grid cell.
                        bh  =  float(obj_bb_height) / float(yolo_interval)
                        bw  =  float(obj_bb_width)  / float(yolo_interval)
                        # You have to be CAREFUL about object center calculation since bounding-box coordinates
                        # are in (x,y) format --- with x-positive going to the right and y-positive going down.
                        obj_center_x =  float(bbox_tensor[ibx][idx][2].item() +  bbox_tensor[ibx][idx][0].item()) / 2.0
                        obj_center_y =  float(bbox_tensor[ibx][idx][3].item() + bbox_tensor[ibx][idx][1].item()) / 2.0

                        # Now you need to switch back from (x,y) format to (i,j) format:
                        yolocell_center_i =  cell_row_idx*yolo_interval + float(yolo_interval) / 2.0
                        yolocell_center_j =  cell_col_idx*yolo_interval + float(yolo_interval) / 2.0
                        # if yolo_debug:
                        #     print("object center located at: obj_center_x: %.3f    obj_center_y: %0.3f" % 
                        #                                                               (obj_center_x, obj_center_y))
                        #     print("yolocell center located at: yolocell_center_j: %.3f   yolocell_center_i: %0.3f" % 
                        #                                                     (yolocell_center_j, yolocell_center_i))
                        del_x = float(obj_center_x - yolocell_center_j) / yolo_interval
                        del_y = float(obj_center_y - yolocell_center_i) / yolo_interval
                        yolo_vector = [1, del_x, del_y, bh, bw, 0, 0, 0]
                        yolo_vector[5 + bbox_label_tensor[ibx][idx].item()] = 1
                        ## Remember because the coordinate reversal between (x,y) and (i,j) formats, cell_row_idx
                        ## is the index along the horizontal dimension and cell_col_idx is along the vertical dimension.
                        yolo_cell_index =  cell_row_idx * num_cells_image_width  +  cell_col_idx
                        # if yolo_debug:
                        #     print("iter=%d  yolo_vector: " % iter, yolo_vector)
                        #     print("iter=%d  yolo_cell_index: " % iter, yolo_cell_index)
                        yolo_tensor[ibx, yolo_cell_index, anchbox.adx] = torch.FloatTensor( yolo_vector )
                        # print(yolo_tensor)
                    bbox_actual.append(bbox_actual_ibx)
                
                yolo_tensor_flattened = yolo_tensor.view(im_tensor.shape[0], -1)
                
                if iter%5==4:    
                    if display_images:
                        print("\n\n\n")
                    current_time = time.perf_counter()
                    
                    if display_images:
                        predictions = output.view(4,36,5,8)
                        # print(predictions[0][0][0])
                        for ibx in range(predictions.shape[0]):                             # for each batch image
                            icx_2_best_anchor_box = {ic : None for ic in range(36)}
                            for icx in range(predictions.shape[1]):                         # for each yolo cell
                                cell_predi = predictions[ibx,icx]
                                # print(f'{icx}: {cell_predi.shape}')
                                prev_best = 0
                                for anchor_bdx in range(cell_predi.shape[0]):
                                    if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                        prev_best = anchor_bdx
                                best_anchor_box_icx = prev_best 
                                # print(best_anchor_box_icx)
                                # print(best_anchor_box_icx)
                                icx_2_best_anchor_box[icx] = best_anchor_box_icx
                            sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                       key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                            retained_cells = sorted_icx_to_box[:3]
                            # print(f'{icx_2_best_anchor_box}')
                            # print(f'{sorted_icx_to_box}')
                            # print(f'{retained_cells}')
                            objects_detected = []
                            bboxes_pred = []
                            cell_center_pred_idx = []
                            cell_center_pred = []
                            bbox_pred_center = []
                            
                            for icx in retained_cells:
                                cell_col_idx_pred = icx % 6 
                                cell_row_idx_pred = icx // 6
                                cell_center_pred_idx.append([cell_row_idx_pred, cell_col_idx_pred])
                                
                                cell_col_pred = cell_col_idx_pred * 20 + 10
                                cell_row_pred = cell_row_idx_pred * 20 + 10
                                cell_center_pred.append([cell_row_pred, cell_col_pred])
                                
                                pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                                # print(f'Prediction Vector for each bbox for image {ibx}: {pred_vec}')
                                
                                dx = pred_vec[1].item()
                                dy = pred_vec[2].item()
                                dw = pred_vec[3].item()
                                dh = pred_vec[4].item()
                                bboxes_pred.append([dx, dy, dw, dh])
                                
                                bbox_x = 20 * dx + cell_col_pred
                                bbox_y = 20 * dy + cell_row_pred
                                bbox_w = 20 * dw
                                bbox_h = 20 * dh
                                
                                bbox_pred_center.append([bbox_x - bbox_w/2, bbox_y - bbox_h/2, bbox_w, bbox_h])
                                
                                
                                class_labels_predi  = pred_vec[-3:]                        
                                
                                if torch.all(class_labels_predi < 0.2): 
                                    predicted_class_label = None
                                else:
                                    best_predicted_class_index = (class_labels_predi == 
                                                                      class_labels_predi.max()).nonzero().squeeze().item()
                                    predicted_class_label = classes[best_predicted_class_index]
                                    objects_detected.append(predicted_class_label)

                            print(bbox_label_tensor[ibx].cpu().numpy(), [classes.index(objects_detected[0]), classes.index(objects_detected[1]), classes.index(objects_detected[2])])
                            for label,prediction in zip(bbox_label_tensor[ibx].cpu().numpy(), [classes.index(objects_detected[0]), classes.index(objects_detected[1]), classes.index(objects_detected[2])]):
                                confusion_matrix[label][prediction] += 1
                                # print(confusion_matrix[label][prediction])
#                                    print("[batch image=%d] most probable object instance detected: ", objects_detected[-1])
#                                    if yolo_debug:
                            # print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                            # print(bboxes_pred)
                            # print(f'cell center index {cell_center_pred_idx}')
                            # print(f'cell center {cell_center_pred}')
                            # print(f'bbox predicted {bbox_pred_center}')
                            # print(f'bbox actual {bbox_tensor}')
                            # print(f'bbox actual {bbox_tensor[0]}')
                            # print(f'bbox actual label {bbox_label_tensor}')
                            print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                            print("[batch image=%d]  objects actual: " % ibx, [classes[bbox_label_tensor[ibx][0]], classes[bbox_label_tensor[ibx][1]], classes[bbox_label_tensor[ibx][2]]])
                            
            
                    # create masks
                    inputs = torch.zeros([3,128,128])
                    targets = torch.zeros([3,128,128])
                    
                    for k in range(3):
                        for i in range(128):
                            for j in range(128):
                                if (j >= bbox_tensor[0][k][0] and j <= bbox_tensor[0][k][0] + bbox_tensor[0][k][2]) and (i >= bbox_tensor[0][k][1] and i <= bbox_tensor[0][k][1] + bbox_tensor[0][k][3]):
                                    inputs[k][i][j] = 1
                                    # print(inputs[i,j])
                                else:
                                    inputs[k][i][j] = 0
                                
                                if (j >= bbox_pred_center[k][0] and j <= bbox_pred_center[k][0] + bbox_pred_center[k][2]) and (i >= bbox_pred_center[k][1] and i <= bbox_pred_center[k][1] + bbox_pred_center[k][3]):
                                    targets[k][i][j] = 1
                                else:
                                    targets[k][i][j] = 0
                    # plt.figure(101)
                    # plt.imshow(inputs[0])
                    # plt.show()
                    # plt.imshow(targets[0])
                    # plt.show()

                    # print(inputs, targets)
                    IoULoss = self.IoULossCalc(inputs, targets)
                    IoULoss = IoULoss.cpu().numpy()
                    print(f'{IoULoss} is the IOU loss')
                    iouLoss.append(IoULoss)
                    
                
                    if display_images:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[150,40])
                        
                        # image = np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True, padding=3, pad_value=255).cpu(), (1,2,0))
                        # image = skimage.color.gray2rgb(im_tensor[0].cpu())
                        # imgg = np.asarray(image)
                        # imgg = np.uint8(imgg)
                        im_tensor = im_tensor[0].cpu()
                        image = im_tensor.numpy()
                        image = image.transpose(1,2,0)
                        fig, ax = plt.subplots(1,1)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        # plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                        # padding=3, pad_value=255).cpu(), (1,2,0)))                        
                        # plt.imshow(np.transpose(im_tensor[0].cpu()))
                        for t in range(3):
                            # plot the bounding box around the image
                            image = cv2.rectangle(image, (int(bbox_pred_center[t][0]), int(bbox_pred_center[t][1])), (int(bbox_pred_center[t][0] + bbox_pred_center[t][2]), int(bbox_pred_center[t][1] + bbox_pred_center[t][3])), (0 ,0 ,255), 1)
                            image = cv2.rectangle(image, (int(bbox_tensor[0][t][0]), int(bbox_tensor[0][t][1])), (int(bbox_tensor[0][t][0] + bbox_tensor[0][t][2]), int(bbox_tensor[0][t][1] + bbox_tensor[0][t][3])), (0 ,255 ,0), 1)

                            if len(objects_detected) == 3:
                            # put a label next to the bounding box
                                image = cv2.putText(image, objects_detected[t], (int(bbox_pred_center[t][0]),int(bbox_pred_center[t][1]-5)), cv2.FONT_HERSHEY_SIMPLEX ,0.3, (0 , 0 ,255), 1)
                                image = cv2.putText(image, classes[bbox_label_tensor[0][t]], (int(bbox_tensor[0][t][0]),int(bbox_tensor[0][t][1]-5)), cv2.FONT_HERSHEY_SIMPLEX ,0.3, (0, 255, 0), 1)
                                
                        
                        plt.imshow(image)
                        logger.setLevel(old_level) 
            
                print(confusion_matrix)
        print("\n\nDisplaying the confusion matrix:\n")
        out_str = "                "
        for j in range(len(classes)):  
                             out_str +=  "%15s" % classes[j]   
        print(out_str + "\n")
        for i,label in enumerate(classes):
            out_percents = [confusion_matrix[i,j] 
                             for j in range(len(classes))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%12s:  " % classes[i]
            for j in range(len(classes)): 
                                                   out_str +=  "%15s" % out_percents[j]
            print(out_str)
       
        print(iouLoss)
        
        plt.figure(100)
        plt.plot(iouLoss)
        plt.show()
       
                # for ibx in range(predictions.shape[0]):                             # for each batch image
                #     icx_2_best_anchor_box = {ic : None for ic in range(36)}
                #     for icx in range(predictions.shape[1]):                         # for each yolo cell
                #         cell_predi = predictions[ibx,icx]               
                #         prev_best = 0
                #         for anchor_bdx in range(cell_predi.shape[0]):
                #             if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                #                 prev_best = anchor_bdx
                #         best_anchor_box_icx = prev_best   
                #         icx_2_best_anchor_box[icx] = best_anchor_box_icx
                #     sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                #                key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                #     retained_cells = sorted_icx_to_box[:5]
                #     objects_detected = []
                #     for icx in retained_cells:
                #         pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]
                #         class_labels_predi  = pred_vec[-3:]                        
                #         if torch.all(class_labels_predi < 0.05): 
                #             predicted_class_label = None
                #         else:
                #             best_predicted_class_index = (class_labels_predi == 
                #                                               class_labels_predi.max()).nonzero().squeeze().item()
                #             predicted_class_label = classes[best_predicted_class_index]
                #             objects_detected.append(predicted_class_label)
                #     print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                
                
                # logger = logging.getLogger()
                # old_level = logger.level
                # logger.setLevel(100)
                # plt.figure(figsize=[15,4])
                # plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                #                                                  padding=3, pad_value=255).cpu(), (1,2,0)))
                # plt.show()
                # logger.setLevel(old_level)
                        

yolo = YoloLikeDetector()

# dataserver_train = PurdueDrEvalMultiDataset("train", dataroot_train=dataroot_train)
# train_dataloader = torch.utils.data.DataLoader(dataserver_train, batch_size, shuffle=True)
yolo.set_dataloaders(test=True)
# dataserver_train.__getitem__()
model = yolo.NetForYolo(skip_connections=True, depth=8) 
number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the model: %d\n\n" % num_layers)

# model = yolo.run_code_for_training_multi_instance_detection(model, display_images=True)

yolo.run_code_for_testing_multi_instance_detection(model, display_images = True)
