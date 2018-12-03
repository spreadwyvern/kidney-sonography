from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt

def plot_roc(tpr, fpr, save_dir):
    plt.figure()  
    roc_auc = auc(fpr, tpr)
    lw = 2  
    plt.figure(figsize=(7,7))  
    plt.plot(fpr, tpr, color='darkorange',  
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating characteristic')  
    plt.legend(loc="lower right") 
    plt.savefig(save_dir + 'auc.png', dpi=300)
    plt.show()  