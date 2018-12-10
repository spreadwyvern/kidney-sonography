import os
import pandas as pd
import cv2
from PIL import Image, ImageOps, ImageEnhance
from utils.model import freeze_blocks_resnet, CNN
from utils.img_preprocess import img_process_transform, img_process_PIL

import xgboost as xgb
import pickle

import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=str, default="0", help='GPU ID')
    # parser.add_argument('-p', '--img_path', type=str, default='', help='img_path')

    FLAG = parser.parse_args()

    predict_kidney(FLAG)

def predict_kidney(FLAG):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAG.gpu_id
    
    torch.cuda.set_device(0)
    # specify dtype
    use_cuda = torch.cuda.is_available()
    print("use GPU: {}".format(use_cuda))
    if use_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print("numbers of GPU: {}".format(torch.cuda.device_count()))
    
    # normalization of image
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    while True:
        img_path = input('Input image path: ')
        try:
            img = img_process_PIL(img_path, None, transformations, False, (224, 224))
            img = Variable(img).type(dtype)
            img = img.unsqueeze(0)
        except:
            print('Unable to open image! Try again!')
            continue
        else:

            if 'cnn_models' not in locals():
                print("loading trained models...")
                # create models dict
                cnn_models = {}
                extract_models = {}
                # model directory 
                model_dir = 'models/'
                # model ensemble
                for i in range(10):    
                    # ResNet-101 in submitted paper
                    model = CNN(drop_rate=0.5).type(dtype)
                    net = torch.nn.DataParallel(model)
                    net.load_state_dict(torch.load(model_dir+ 'model_{}.pkl'.format(i)))

                    extract_CNN = nn.Sequential(*list(net.module.base_model.children())[:-1]).type(dtype)
                    extract_net = torch.nn.DataParallel(extract_CNN)
                    print("loaded model {}/10".format(i), end = '\r')
                    cnn_models["model_{}".format(i)] = net
                    extract_models["extract_model_{}".format(i)] = extract_net
                    del net
                print("loaded trained CNN models")
            else:
                print("CNN models already loaded")
            
            if 'xgb_models' not in locals():
                print("loading trained XGBoost models...")

                xgb_models = {}
                xgb_dir = 'models/'
                for i in range(10):
                    xgb_model = pickle.load(open(xgb_dir + 'xgb_{}.pickle.dat'.format(i), "rb"))
                    xgb_models["xgb_{}".format(i)] = xgb_model
                print("loaded trained XGBoost models")
            else:
                print("XGBoost models already loaded")

            egfr = predict_egfr(img, cnn_models)
            print("Predictred eGFR: {}".format(egfr))
            ckd_stage = predict_ckd(img, extract_models, xgb_models)
            print("Predicted CKD stage < III: {:.2f}%".format(ckd_stage*100))
                    

def predict_egfr(input_img, cnn_models):
    pred = []
    
    print("predicting egfr...")    
    for i in range(10):
        test_pred = cnn_models["model_{}".format(i)].eval()(input_img)
        test_pred = test_pred.data.cpu().numpy()
        pred.append(test_pred)
        print("prediction {}/10 completed".format(i+1), end="\r")


    ens_pred = sum(pred)/len(pred)


    return(ens_pred)

def predict_ckd(input_img, extract_models, xgb_models):
    pred = []

    print("predicting CKD stage...")
    for i in range(10):
        test_features = extract_models["extract_model_{}".format(i)].eval()(input_img)
        test_features = test_features.data.cpu().numpy()
        test_features = pd.DataFrame(test_features.reshape(-1, 2048))
        test_features.columns = [str(i) for i in range(2048)]
        test_features = test_features[xgb_models["xgb_{}".format(i)].feature_names]
        test_features = xgb.DMatrix(test_features)
        test_class = xgb_models["xgb_{}".format(i)].predict(test_features)
        pred.append(test_class)
        print("prediction {}/10 completed".format(i+1), end="\r")

    ens_pred = sum(pred)/len(pred)
    # ens_pred = ens_pred > 0.5

    return(ens_pred[0])

if __name__ == '__main__':
    main()


