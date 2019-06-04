import numpy as np
import itertools
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import auc
import matplotlib.font_manager as font_manager

fontsize_ax = 12
font = font_manager.FontProperties(family='Times New Roman', size=fontsize_ax)
font_name = "Times New Roman"
class PlotStats():
    def __init__(self):
        plt.figure()
        plt.rcParams['figure.figsize'] = [10, 8]
        
    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        list_ = ['16QAM', '8QAM', 'QPSK', 'None', ]
        plt.xticks(tick_marks, list_, fontsize=10, rotation=45, fontname=font_name)
        plt.yticks(tick_marks, list_, fontsize=10, fontname=font_name)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center", fontsize=fontsize_ax, fontname=font_name,
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=fontsize_ax, fontname=font_name)
        plt.xlabel('Predicted label', fontsize=fontsize_ax, fontname=font_name)
        plt.tight_layout()
        
        plt.show()
        
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of QoT-E (1)')
        plt.legend(loc="lower right")
        plt.show()
        
    def plot_roc_curve_multi(self, fpr, tpr, roc_auc, n_classes):
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            
        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (AUC = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (AUC = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        lw = 2
        plt.plot(fpr[0], tpr[0], color='turquoise', lw=lw,
                     label='ROC curve of OSNR >= 17 dB (AUC = {1:0.2f})'
                     ''.format(0, roc_auc[0]))
        plt.plot(fpr[1], tpr[1], color='lightsalmon', lw=lw,
                     label='ROC curve of OSNR >= 14 dB (AUC = {1:0.2f})'
                     ''.format(1, roc_auc[1]))
        plt.plot(fpr[2], tpr[2], color='goldenrod', lw=lw,
                     label='ROC curve of OSNR >= 10 dB (AUC = {1:0.2f})'
                     ''.format(2, roc_auc[2]))
        plt.plot(fpr[3], tpr[3], color='darkviolet', lw=lw,
                     label='ROC curve of OSNR < 10 dB (AUC = {1:0.2f})'
                     ''.format(3, roc_auc[3]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=fontsize_ax, fontname=font_name)
        plt.yticks(fontsize=fontsize_ax, fontname=font_name)
        plt.xlabel('False Positive Rate', fontsize=fontsize_ax, fontname=font_name)
        plt.ylabel('True Positive Rate', fontsize=fontsize_ax, fontname=font_name)
        plt.title('Receiver operating characteristic to multi-class QoT-E')
        plt.legend(loc="lower right", prop=font, fontsize=10)
        plt.show()
