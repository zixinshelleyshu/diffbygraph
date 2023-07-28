import os
import pandas as pd
import random
import torch
import pandas as pd
from model import DenseNet121
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from data_loading import Load_from_path_Dataset


#Training settings
parser = argparse.ArgumentParser(description='graph visual')
parser.add_argument('--name', type=str, default="all training densenet pretrain unweighted bce" )
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=38, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--imgw', type=int, default=320)
parser.add_argument('--imgh', type=int, default=320)
parser.add_argument('--bs', type=int, default=32, help="batch_size")
parser.add_argument('--nc', type=int, default=5, help="number of classes")
parser.add_argument('--interp', type=str, default="gc", help="interpretability method")
parser.add_argument('--select_short', type=str, default=False, help="selecting smaller portion of image for debugging")
parser.add_argument('--n_epochs_stop',type=float, default=12, help="the number of epoch waiting before early stopping")
parser.add_argument('--location', type=str)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

short_select=100
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# torch.use_deterministic_algorithms(True)

homepath="/home/shelley/Documents/CheXpert/"
train_labels_meta=pd.read_csv(homepath+"CheXpert-v1.0/withtestset_alltrain/train.csv")
val_labels_meta=pd.read_csv(homepath+"CheXpert-v1.0/withtestset_alltrain/val.csv")
test_labels_meta=pd.read_csv(homepath+"CheXpert-v1.0/withtestset_alltrain/test.csv")

print("training data:", train_labels_meta.shape[0])
print("validation data", val_labels_meta.shape[0])
print("test_data", test_labels_meta.shape[0])

exist_labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

y_train=train_labels_meta[exist_labels].values
y_val=val_labels_meta[exist_labels].values
y_test=test_labels_meta[exist_labels].values

x_train_path=train_labels_meta.Path
x_val_path=val_labels_meta.Path
x_test_path=test_labels_meta.Path


if args.select_short:
    train_dataset=Load_from_path_Dataset(x_train_path[:short_select], homepath, y_train[:short_select],args.imgw, args.imgh)
    val_dataset=Load_from_path_Dataset(x_val_path, homepath, y_val,args.imgw, args.imgh)
    test_dataset=Load_from_path_Dataset(x_test_path, homepath+"CheXpert-v1.0/", y_test, args.imgw, args.imgh)
    print("selected less training data for debugging")

else:
    train_dataset=Load_from_path_Dataset(x_train_path, homepath, y_train,args.imgw, args.imgh)
    val_dataset=Load_from_path_Dataset(x_val_path, homepath, y_val,args.imgw, args.imgh)
    test_dataset=Load_from_path_Dataset(x_test_path, homepath+"CheXpert-v1.0/", y_test, args.imgw, args.imgh)


train_dataloader=DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
val_dataloader=DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
test_dataloader=DataLoader(test_dataset, batch_size=args.bs, shuffle=False)



def train(epoch,train_dataloader, model, optimizer, writer,step, n_classes):
    model.train()
    if args.cuda:
        model.cuda()

    train_loss=[]
    labels_list=[]
    preds_list=[]
        
    for batch_indx, data_batch in enumerate(train_dataloader):
        image, labels=data_batch[0].float(),data_batch[1].float()

        # calculating class weight
        class_proportion=1-(torch.sum(labels, 0)/labels.shape[0])
        class_weights=class_proportion/class_proportion.sum()

        if args.cuda:
            image, labels, class_weights=image.cuda(), labels.cuda(), class_weights.cuda()

        optimizer.zero_grad()
        batch_norm=torch.nn.BatchNorm2d(3, affine=False).cuda()
        image=batch_norm(image)
        preds=model(image)

        # criterion = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=class_weights.data)
        criterion = torch.nn.BCELoss()
        loss=criterion(preds,target=labels)

        # criterion1 = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.data)
        # loss1=criterion1(preds,target=labels)
        
        labels_list.append(labels)
        preds_list.append(preds)
        train_loss.append(loss.cpu().data.numpy())

        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss', loss, step)
        step+=1

    mean_train_loss=np.mean(train_loss)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch,mean_train_loss))

    targets=torch.cat(labels_list).cpu().data.numpy()
    outputs=torch.cat(preds_list).cpu().data.numpy()
    roc_auc=roc_auc_score(targets, outputs, average=None)
    print("auroc", roc_auc)

    return mean_train_loss, step


def val(writer, step_val, valid_dataloader,model, n_classes):
    model.eval()

    val_loss_list=[]
    labels_list=[]
    preds_list=[]

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(valid_dataloader):
            image, labels=data_batch[0].float(),data_batch[1].float()

            # calculating class weight
            class_proportion=1-(torch.sum(labels, 0)/labels.shape[0])
            class_weights=class_proportion/class_proportion.sum()

            if args.cuda:
                image, labels, class_weights=image.cuda(), labels.cuda(), class_weights.cuda()

            batch_norm=torch.nn.BatchNorm2d(3, affine=False).cuda()
            preds=model(batch_norm(image))

            # criterion = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=class_weights.data)
            criterion = torch.nn.BCELoss()
            loss=criterion(preds,target=labels)

            val_loss_list.append(loss.cpu().data.numpy())
            writer.add_scalar('validation loss',loss, step_val)

            labels_list.append(labels.cpu().data.numpy())
            preds_list.append(preds.cpu().data.numpy())

            step_val+=1


        val_loss=np.mean(val_loss_list)
        print("validation loss", val_loss)

        targets=np.concatenate(labels_list)
        outputs=np.concatenate(preds_list)

        roc_auc=roc_auc_score(targets, outputs, average=None)
        ap=average_precision_score(targets, outputs)

        print("auroc", roc_auc)
        print("ap",round(ap,4))

        return val_loss, step_val,np.mean(roc_auc)


    
def test(test_dataloader, n_classes,model):
    model.eval()

    test_loss=[]
    labels_list=[]
    preds_list=[]

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(test_dataloader):
            image, labels=data_batch[0].float(),data_batch[1].float()
            # calculating class weight
            class_proportion=1-(torch.sum(labels, 0)/labels.shape[0])
            class_weights=class_proportion/class_proportion.sum()

            if args.cuda:
                image, labels, class_weights=image.cuda(), labels.cuda(), class_weights.cuda()

            batch_norm=torch.nn.BatchNorm2d(3, affine=False).cuda()
            preds=model(batch_norm(image))

            # criterion = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=class_weights.data)
            criterion = torch.nn.BCELoss()
            loss=criterion(preds,target=labels)

            labels_list.append(labels.cpu().data.numpy())
            preds_list.append(preds.cpu().data.numpy())
            test_loss.append(loss.cpu().data.numpy())

        test_loss_mean=np.mean(test_loss)
        print("test_loss", test_loss_mean)

        targets=np.concatenate(labels_list)
        outputs=np.concatenate(preds_list)

        roc_auc=roc_auc_score(targets, outputs)
        ap=average_precision_score(targets, outputs)

        return roc_auc, ap, test_loss_mean
    


if __name__ == "__main__":
    storinghome="/home/shelley/Documents/diffbygraph"
    PATH_TB=storinghome+'/tensorboard_results'
    os.makedirs(PATH_TB,exist_ok=True)
    os.makedirs(PATH_TB+"/"+"seed"+str(args.seed),exist_ok=True)

    writer=SummaryWriter(PATH_TB+"/seed"+str(args.seed)+"/tensorboard_logs_{}_{}_lr{}_epoches{}".format(args.name,args.interp,args.lr, args.epochs))

    step_val=0
    step=0
    max_auc=0
    model=DenseNet121(num_classes=args.nc)
    optimizer=optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, f'min', patience=5)


    for epoch in range(args.epochs):
        training_loss, step=train(epoch=epoch, train_dataloader=train_dataloader, model=model, optimizer=optimizer, writer=writer, step=step, n_classes=args.nc)
        val_loss, step_val,roc_auc=val(writer=writer, step_val=step_val, model=model,valid_dataloader=val_dataloader, n_classes=args.nc)
        scheduler.step(val_loss)

        #early stopping
        if (roc_auc-max_auc)>0.001:
            best_model=model
            max_auc=roc_auc
            epoch_min=epoch
            epochs_no_improve=0

            model_state=model.state_dict()
            optimizer_state=optimizer.state_dict()

        else:
            epochs_no_improve += 1

        if epoch > 10 and epochs_no_improve ==args.n_epochs_stop:
            print('Early stopping!' )
            break
        else:
            continue

    writer.close()

    # saving the model
    PATH_SAVE=storinghome+'/models'+"/seed"+str(args.seed)
    os.makedirs(PATH_SAVE, exist_ok=True)
    PATH_SAVE_MODEL=PATH_SAVE+"/{}_model_{}_lr{}_epoches{}.pt".format(args.name, args.interp,args.lr, args.epochs)
    torch.save({'epoch':epoch_min, 'model_state_dict': model_state,'optimizer_state_dict': optimizer_state,'AUC':max_auc}, PATH_SAVE_MODEL)

    print('Start Testing')
    roc_auc, ap, test_loss=test(test_dataloader=test_dataloader,n_classes=args.nc,model=best_model)
    print("test roc_auc:", roc_auc, "average precision:", ap,"test_loss",test_loss)


    # outputing results
    RESULT_SAVING=storinghome+'/evalution_metrices'
    os.makedirs(RESULT_SAVING,exist_ok=True)

    os.makedirs(RESULT_SAVING+"/seed"+str(args.seed),exist_ok=True)
    file_name="{}_evaluation_metries_{}_lr{}_epoches{}.csv".format(args.name, args.interp, args.lr, args.epochs)
    file_path=RESULT_SAVING+"/seed"+str(args.seed)+"/"+file_name
    with open(file_path, 'w') as csvfile:
        writer=csv.DictWriter(csvfile, fieldnames=["roc_auc","average precision" ,"accuracy", "test_loss"])
        writer.writeheader()
        writer.writerows([{"roc_auc":roc_auc, "average precision":ap, "test_loss":test_loss}])













        









    
    







