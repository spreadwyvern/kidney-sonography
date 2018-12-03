
# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler, TensorDataset, Dataset
import torch.utils.model_zoo as model_zoo

from sklearn import metrics
import itertools
import math
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import pandas as pd
from pandas import Series
import random
import scipy
import time
import cv2
from PIL import Image, ImageOps, ImageEnhance

from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from utils.plot import plot_losses, plot_making, relative_error, loss_generator
from utils.generator import kidney_Dataset
from utils.model import freeze_blocks_resnet, CNN, Inception_v4, CNN_VGG
from utils.img_preprocess import img_process_transform, img_process_PIL

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('-p', '--img_path', type=str, default='', help='img_path')

    FLAG = parser.parse.args()

    predict_egfr(FLAG)



def predict_egfr(FLAG):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAG.gpu_id    
     # normalization of image
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    while True:

        try:
            img = img_process_PIL(FLAG.img_path, None, transformations, None, (224, 224))
        except:
            print('Unable to open image! Try again!')
            continue
        else:
            egfr = predict(img)
            print(egfr)        

def predict(input_img):

    torch.cuda.set_device(0)
    # specify dtype
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if use_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print(torch.cuda.device_count())
    
    input_img = Variable(input_img, volatile=True).type(dtype)
    input_img = input_img.unsqueeze(0)

    drop_rate = 0.5
    true = []
    pred = []
    
    if 'models' not in globals():
        print("loading trained models...")
        # create models dict
        models = {}

        # model ensemble
        for i in range(10):    
            # ResNet-101 in submitted paper
            model = CNN(drop_rate=0.5).type(dtype)
            net = torch.nn.DataParallel(model)

            # model directory 
            model_dir = '/models/'

            net.load_state_dict(torch.load(model_dir+ 'model_{}.pkl'.format(i)))
            print("loaded model {}".format(i))
            models["model{}".format(i)] = net
        print("loaded trained models")

    print("predicting...")    
    for i in range(10):
        test_pred = models["model{}".format(i)](input_img)
        print("prediction {}/10 completed".format(i+1))
        del net
        del model

    pred2 = sum(pred)/len(pred)


    # In[ ]:
    return(pred2)

if __name__ == '__main__':
    main()


