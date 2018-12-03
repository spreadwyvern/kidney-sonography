# '''import packages'''
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision
# import torchvision.transforms as transforms
import torchvision.models as models
# from torch.utils.data import sampler, TensorDataset, Dataset
# import torch.utils.model_zoo as model_zoo
import numpy as np


'''freeze blocks for resnet'''
def freeze_blocks_resnet(model, blocks):
    if blocks == 0:
        pass
    elif blocks == 1:        
        for param in model.base_model.layer1.parameters():
            param.requires_grad = False
    else:
        for param in model.base_model.layer2.parameters():
            param.requires_grad = False
    return model

'''architecture'''
class CNN(nn.Module):
    def __init__(self, drop_rate, pretrained=True):
        super(CNN, self).__init__()
        self.base_model = models.resnet101(pretrained=pretrained)
        self.flat_d = self.get_flat_d(Variable(torch.ones(1, 3, 224, 224)), self.base_model)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_d, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1)
        )
        
    def get_flat_d(self, x, conv):
        out = conv(x)
        return int(np.prod(out.size()[1:]))
    
    def forward(self, x1):
        out2 = self.base_model(x1)
        out2 = out2.view(out2.size(0), -1)
        out = self.fc(out2)
        return out