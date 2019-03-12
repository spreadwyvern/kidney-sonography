import os
import pandas as pd
import cv2
from PIL import Image, ImageOps, ImageEnhance
from utils.model import freeze_blocks_resnet, CNN
from utils.img_preprocess import img_process_transform, img_process_PIL

import xgboost as xgb
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models


class Model:
    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.fileName = None
        self.fileContent = ""

    def isValid(self, fileName):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        if fileName.endswith('.png'):
            try:
                file = open(fileName, 'r')
                file.close()
                return True
            except:
                return False
        else:
            return False


    def setFileName(self, fileName):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid(fileName):
            self.fileName = fileName
            # self.fileContents = open(fileName, 'r').read()
        else:
            # self.fileContents = ""
            self.fileName = ""

    def getFileName(self):
        '''
        Returns the name of the file name member.
        '''
        return self.fileName

    def getFileContents(self):
        '''
        Returns the contents of the file if it exists, otherwise
        returns an empty string.
        '''
        return self.fileContents


    def load_img(self, path):
        img_path = path
        # normalization of image
        transformations = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        img = img_process_PIL(img_path, None, transformations, False, (224, 224))
        img = Variable(img).type(self.dtype)
        img = img.unsqueeze(0)

        return img

    def load_cnn(self, progress_bar):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
        # specify dtype
        use_cuda = torch.cuda.is_available()
        # print("use GPU: {}".format(use_cuda))
        if use_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        # print("numbers of GPU: {}".format(torch.cuda.device_count()))
        cnn_models = {}
        extract_models = {}
        # model directory
        model_dir = 'models/'
        # model ensemble
        for i in range(10):
            # ResNet-101 in submitted paper
            model = CNN(drop_rate=0.5).type(self.dtype)
            net = torch.nn.DataParallel(model)
            net.load_state_dict(torch.load(model_dir + 'model_{}.pkl'.format(i)))

            extract_CNN = nn.Sequential(*list(net.module.base_model.children())[:-1]).type(self.dtype)
            extract_net = torch.nn.DataParallel(extract_CNN)
            # print("loaded model {}/10".format(i), end='\r')
            cnn_models["model_{}".format(i)] = net
            extract_models["extract_model_{}".format(i)] = extract_net
            del net
            progress_bar.emit(5)
        return cnn_models, extract_models

    def load_xgb(self, progress_bar):
        xgb_models = {}
        xgb_dir = 'models/'
        for i in range(10):
            xgb_model = pickle.load(open(xgb_dir + 'xgb_{}.pickle.dat'.format(i), "rb"))
            xgb_models["xgb_{}".format(i)] = xgb_model
            progress_bar.emit(5)
        # print("loaded trained XGBoost models")
        return xgb_models

    def predict_kidney(self, img, cnn_models, extract_models, xgb_models):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = FLAG.gpu_id

        egfr = self.predict_egfr(img, cnn_models)
        # print("Predictred eGFR: {}".format(egfr))
        ckd_stage = self.predict_ckd(img, extract_models, xgb_models)
        # print("Predicted CKD stage > II: {:.2f}%".format(ckd_stage * 100))

        return egfr, ckd_stage*100

    def predict_egfr(self, input_img, cnn_models, progress_bar):
        pred = []

        # print("predicting egfr...")
        for i in range(10):
            test_pred = cnn_models["model_{}".format(i)].eval()(input_img)
            test_pred = test_pred.data.cpu().numpy()
            pred.append(test_pred)
            print("prediction {}/10 completed".format(i + 1), end="\r")
            progress_bar.emit(5)

        ens_pred = sum(pred) / len(pred)

        return (ens_pred)


    def predict_ckd(self, input_img, extract_models, xgb_models, progress_bar):
        pred = []

        # print("predicting CKD stage...")
        for i in range(10):
            test_features = extract_models["extract_model_{}".format(i)].eval()(input_img)
            test_features = test_features.data.cpu().numpy()
            test_features = pd.DataFrame(test_features.reshape(-1, 2048))
            test_features.columns = [str(i) for i in range(2048)]
            test_features = test_features[xgb_models["xgb_{}".format(i)].feature_names]
            test_features = xgb.DMatrix(test_features)
            test_class = xgb_models["xgb_{}".format(i)].predict(test_features)
            pred.append(test_class)
            # print("prediction {}/10 completed".format(i + 1), end="\r")
            progress_bar.emit(5)

        ens_pred = sum(pred) / len(pred)
        # ens_pred = ens_pred > 0.5

        return (ens_pred[0])


