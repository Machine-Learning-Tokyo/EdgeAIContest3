# -*- coding: utf-8 -*-
# -----------------------------------------------------------
#  Generate the training data from annotations for object detection training
#  Released under Apache License 2.0
#  Email: machine.learning.tokyo@gmail.com
# -----------------------------------------------------------
import json
import pdb
import os
import numpy as np

import cv2

# change this path to correct train_annotations folder provided from signate
ann_folder = './train_annotations/'

# change this path to where you want to generate the annotated csv file
csv_ann = './retinanet_annotations.csv.train.all_frames.all_objects.5_classes'

csv_ann_file = open(csv_ann, 'w')

# list up all the training annotation files
json_files = os.listdir(ann_folder)

for json_file in json_files:
    if json_file.split('.')[-1] != 'json':
        continue

    if json_file in ['train_00.json', 'train_01.json', 'train_02.json']:
        continue
    with open(ann_folder + json_file, 'r') as d:
        data = json.load(d)
        anns = data['sequence']
        for i, ann in enumerate(anns):
            impath = './train_videos/'+json_file.split('.')[0] + '/' + str(i+1).zfill(3) + '.png'
            for object_name in ann.keys():
                if object_name not in ['Pedestrian', 'Car', 'Truck', 'Bus', 'Svehicle']:
                    continue
                object_ann = ann[object_name]
                for instants in object_ann:
                    [x1,y1,x2,y2] =  instants['box2d']
                    x1 = max(0,x1)
                    y1 = max(0,y1)
                    x2 = x2 - 3
                    y2 = y2 - 3
                    w = x2 - x1
                    h = y2 - y1
                    if y2 < y1+5 or x2 < x1+5:
                        continue

                    line = impath + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + object_name + '\n'
                    csv_ann_file.write(line)



