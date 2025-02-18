import json
import os
import shutil
from src.utils.load_env import ENV      # import your machine environment
from src.gateway.dataset import *

def format_transform(dataset_name,partition_name):
    dataset_dir = os.path.join(ENV["data_root_dir"], dataset_name)
    annotation_dir = os.path.join(dataset_dir, "annotations", partition_name)
    train_annotation_path = os.path.join(annotation_dir, "train_annotation.json")
    test_annotation_path = os.path.join(annotation_dir, "test_annotation.json")
    with open(train_annotation_path, 'r') as f:
        train_annotation = json.load(f)
    with open(test_annotation_path, 'r') as f:
        test_annotation = json.load(f)
    target_dir=os.path.join(ENV["data_root_dir"],dataset_name,"processed",partition_name,'abnormal')
    if os.path.exists(target_dir+'/train/Normal'):
        print('dataset already exist.Will overwrite')
        
    else:
        os.makedirs(target_dir+'/train/Normal')
        os.makedirs(target_dir+'/train/Defective')
        os.makedirs(target_dir+'/test/Normal')
        os.makedirs(target_dir+'/test/Defective')

    train_image_names=train_annotation['image_annotation'][0]["image_name"]
    test_image_names=test_annotation['image_annotation'][0]["image_name"]

    train_image_labels=[get_image_annotations(dataset_name,os.path.splitext(os.path.basename(name))[0])['annotations']['classification'][0] for name in train_image_names]
    test_image_labels=[get_image_annotations(dataset_name,os.path.splitext(os.path.basename(name))[0])['annotations']['classification'][0] for name in test_image_names]

    for i in range(len(train_image_names)):
        shutil.copy(os.path.join(dataset_dir,'data','images',train_image_names[i]),os.path.join(target_dir,'train',train_image_labels[i],train_image_names[i]))
    for i in range(len(test_image_names)):
        shutil.copy(os.path.join(dataset_dir,'data','images',test_image_names[i]),os.path.join(target_dir,'test',test_image_labels[i],test_image_names[i]))  
