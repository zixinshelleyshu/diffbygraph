import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve,auc, accuracy_score
import matplotlib.pyplot as plt
import os
def test_evaluate(test_dataloader, n_classes,model,args,exist_labels):
    model.eval()
    test_loss=[]
    labels_list=[]
    preds_list=[]

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(test_dataloader):
            image, labels=data_batch[0].float(),data_batch[1].float()
            
            if args.cuda:
                image, labels,model=image.cuda(), labels.cuda(),model.cuda()

            preds=model(image)

            # criterion = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=class_weights.data)
            criterion = torch.nn.BCELoss()
            loss=criterion(preds,target=labels)

            labels_list.append(labels.cpu().data.numpy())
            preds_list.append(preds.cpu().data.numpy())
            test_loss.append(loss.cpu().data.numpy())

        targets=np.concatenate(labels_list)
        outputs=np.concatenate(preds_list)
        for i in range(args.nc):
            fpr, tpr, thresholds = roc_curve(targets[:,i], outputs[:,i])
            #create ROC curve
            plt.plot(fpr,tpr, label=exist_labels[i])
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.axhline(y=0.8, linestyle='--')
        fig_path="/home/shelley/Documents/diffbygraph/ROCplots/seed{}/".format(str(args.seed))
        os.makedirs(fig_path,exist_ok=True)
        plt.legend()
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.savefig(fig_path+"{}_ROC_lr{}_epoches{}.png".format(args.name, args.lr, args.epochs))
        plt.close()
        roc_auc=roc_auc_score(targets, outputs)
        roc_auc_byclass=roc_auc_score(targets, outputs,average=None)

        return roc_auc, roc_auc_byclass