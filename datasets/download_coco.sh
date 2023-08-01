#!/bin/bash
# Run script with command: bash download_coco.sh

# Make Coco root folder
mkdir coco
cd coco

# Download the 2017 Train images [118K/18GB]
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

# Download the 2017 Val images [5K/1GB]
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# Download the 2017 Test images [41K/6GB]
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
rm test2017.zip

# Download the 2017 Train/Val annotations [241MB]
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

# Download the 2017 Panoptic Train/Val annotations [821MB]
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
rm panoptic_annotations_trainval2017.zip
cd annotations
mv panoptic_train2017.zip panoptic_val2017.zip ../
cd ..
unzip panoptic_train2017.zip
rm panoptic_train2017.zip
unzip panoptic_val2017.zip
rm panoptic_val2017.zip