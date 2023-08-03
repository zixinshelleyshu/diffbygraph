import os
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import PIL
import argparse
from model import DenseNet121
import torch.optim as optim
from data_loading import Load_from_path_Dataset
from torch.utils.data import DataLoader
from captum.attr import LayerGradCam
from utils import plot_similarity_ingraph, compute_similarity 
from torchvision.transforms import Resize
from grad_cam import grad_cam
from sklearn.metrics import roc_auc_score
from evaluate_function import test_evaluate
#settings
parser = argparse.ArgumentParser(description='image image similarity graph')
parser.add_argument('--name', type=str, default="alltrain pretrain unweighted bce gt1fortestval" )
parser.add_argument('--epochs', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=38, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--imgw', type=int, default=320)
parser.add_argument('--imgh', type=int, default=320)
parser.add_argument('--interp', type=str, default="gc", help="interpretability method")
parser.add_argument('--bs', type=int, default=32, help="batch_size")
parser.add_argument('--nc', type=int, default=5, help="number of classes")
parser.add_argument('--patient_type', type=str, default="single")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

PATH_LOAD='/home/shelley/Documents/diffbygraph/models'+"/seed"+str(args.seed)
PATH_LOAD_MODEL=PATH_LOAD+"/{}_model_{}_lr{}_epoches{}.pt".format(args.name, args.interp,args.lr, args.epochs)

model=DenseNet121(num_classes=args.nc)
optimizer=optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
checkpoint = torch.load(PATH_LOAD_MODEL)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
max_auc = checkpoint['AUC']
print('We use the model in traing epoch', epoch,"AUC was", max_auc)

#get the test data for the specific fold
homepath="/home/shelley/Documents/CheXpert/"
test_labels_meta=pd.read_csv(homepath+"CheXpert-v1.0/withtestset_alltrain_gt1fortestval/test.csv")
exist_labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
y_test_all=test_labels_meta[exist_labels].values
x_test_path=test_labels_meta.Path.values   
test_dataset=Load_from_path_Dataset(x_test_path, homepath+"CheXpert-v1.0/", y_test_all, args.imgw, args.imgh)
test_dataloader=DataLoader(test_dataset, batch_size=args.bs, shuffle=False)    
roc_auc, roc_auc_byclass=test_evaluate(test_dataloader=test_dataloader,n_classes=args.nc,model=model, args=args, exist_labels=exist_labels)
print("rerun test evalute ROC:",roc_auc)
print("roc by class", roc_auc_byclass)



if args.patient_type=="single":
    # single disease
    num_disease=y_test_all.sum(axis=1)
    single_disease_patients=[i for i, num in enumerate(num_disease) if num==1]
    single_disease_patient_meta=test_labels_meta.iloc[single_disease_patients,:]
    y_test=single_disease_patient_meta[exist_labels].values
    x_test_path=single_disease_patient_meta.Path.values
    test_dataset=Load_from_path_Dataset(x_test_path, homepath+"CheXpert-v1.0/", y_test, args.imgw, args.imgh)
    test_dataloader_selected=DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

else:
    # multiple diseases 
    num_disease=y_test_all.sum(axis=1)
    multiple_disease_patients=[i for i, num in enumerate(num_disease) if num>1]
    multiple_disease_patient_meta=test_labels_meta.iloc[ multiple_disease_patients,:]
    y_test= multiple_disease_patient_meta[exist_labels].values
    x_test_path=multiple_disease_patient_meta.Path.values
    test_dataset=Load_from_path_Dataset(x_test_path, homepath+"CheXpert-v1.0/", y_test, args.imgw, args.imgh)
    test_dataloader_selected=DataLoader(test_dataset, batch_size=args.bs, shuffle=False)


model.eval()
batch=next(iter(test_dataloader_selected))
image, labels=batch[0].float(),batch[1].float()

    
with torch.no_grad():
    if torch.cuda.is_available():
        image,labels=image.cuda(), labels.cuda()
        model.cuda()
    preds=model(image)

    grad_cam_hooks = {'forward': model.densenet121.features.norm5, 'backward': model.densenet121.classifier[0]}

    # cam = LayerGradCam(model, layer=model.densenet121.features)
    # cam=grad_cam(model, image, grad_cam_hooks,cls_idx=torch.LongTensor([0] * args.bs).cuda())
    attr_classes=[grad_cam(model, image, grad_cam_hooks,cls_idx=torch.LongTensor([i] * args.bs).cuda()).cpu().numpy() for i in range(args.nc)]
    # attr_classes = [torch.Tensor(cam.attribute(image, [i] * image.shape[0])).detach().cpu().numpy() for i in range(args.nc)]
    # get the class map embedding by reinjecting to the model 
    classmap_embeddings=[]
    for i in range(len(attr_classes)):
        class_maps=attr_classes[i]

        resizing=Resize(size=(args.imgh, args.imgw))
        class_maps=resizing(torch.tensor(class_maps))
        class_maps_3channels=torch.zeros((class_maps.shape[0],3,args.imgh, args.imgw))
        class_maps_3channels[:,0,:,:]=torch.squeeze(class_maps,dim=1)
        class_maps_3channels[:,1,:,:]=torch.squeeze(class_maps,dim=1)
        class_maps_3channels[:,2,:,:]=torch.squeeze(class_maps,dim=1)
        classmap_embedding=model.densenet121.features(class_maps_3channels.cuda())
        classmap_embeddings.append(classmap_embedding.cpu().numpy())


    classes_map_bypatient=np.transpose(np.array(classmap_embeddings), (1,0,2,3,4))
    attr_classes_bypatient=np.transpose(np.array(attr_classes), (1,0,2,3,4))
    for p in range(attr_classes_bypatient.shape[0]):
        patients_maps=attr_classes_bypatient[p]
        patient_label=labels[p]
        
        patient_disease=[exist_labels[i] for i, label in enumerate(patient_label) if label==1]

        patient_pred=preds[p]

        Y_hat = torch.ge(patient_pred, torch.tensor([0.244,0.029,0.074,0.674,0.594]).cuda()).float()
        pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
        
        plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                patient_ind=p, patient_disease=patient_disease,preds=pred_disease)
        
        saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/".format(args.name, args.interp, args.lr, args.epochs)
        os.makedirs(saving_graph_path,exist_ok=True)
        plt.savefig(saving_graph_path+"patient{}.png".format(p))










