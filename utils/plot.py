import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.autograd import Variable
import torch

torch.cuda.set_device(0)
# specify dtype
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

'''plot loss curve'''
def plot_losses(loss_history1, loss_history2, model_dir):
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if loss_history1:
        ax1.plot(loss_history1, color="blue", label="train")
    if loss_history2:
        ax1.plot(loss_history2, color="orange", label="test")
    plt.xlabel("epoch") 
    plt.ylabel("loss") 
    plt.legend(bbox_to_anchor=(0.8, 0.9), loc=2, borderaxespad=0.)
    plt.title("loss")
    plt.savefig(model_dir + 'output_losses.png')
    plt.show()
    plt.close()
    
'''scatter plot of target & pred'''   
def plot_making(true, pred, types, model_dir):
    cor = pearsonr(true, pred)[0]
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred) 
    plt.figure(0)
    plt.scatter(true, pred, alpha = .15, s = 20)
    plt.xlabel('True_Y')
    plt.ylabel('Pred_Y')
    plt.title(" data \n" + "MAE = %4f; Cor = %4f; R2 = %4f; #samples = %d" % (mae, cor, r2, len(true)))
    plt.savefig(model_dir + types + "_plot_scatter.png" , dpi = 200)
    plt.show()
    plt.close()

'''calculate relative error for each sample'''
def relative_error(true, pred):
    return np.abs(true - pred) / true    

'''generate plot over all data'''
def loss_generator(testloader, model, types, model_dir, criterion):
    test_loss = []
    true = []
    pred = []
    for i, tdata in enumerate(testloader, 0):
        tinputs1, tlabels = tdata
        tinputs1, tlabels = Variable(tinputs1, volatile=True).type(dtype),Variable(tlabels, volatile=True).type(dtype)
        toutput = model(tinputs1)
        tloss = criterion(toutput, tlabels)
        test_loss.append(tloss.data[0])
        true.extend(tlabels.data.cpu().numpy())
        pred.extend(toutput.data.cpu().numpy())
    true = np.asarray(true)
    true = np.expand_dims(true, axis=1)
    pred = np.asarray(pred)
    re = np.mean(relative_error(true, pred))
    cor = pearsonr(true, pred)[0]
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    plot_making(true, pred, types, model_dir)
    print('L1 Loss on images: %r' % (np.sum(test_loss)/pred.shape[0]))
    return re, cor[0], mae, r2