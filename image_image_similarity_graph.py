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
import seaborn as sns

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
parser.add_argument('--bs', type=int, default=48, help="batch_size")
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
print('We use the model in training epoch', epoch,"AUC was", max_auc)

#get the test data
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
count_correct1=0
count_correct2=0
count_correct3=0
count_correct4=0
count_correct5=0
count_wrong=0
count_correct=0

loss_nornomalised_list_correct=[]
loss_nornomalised_list_0=[]
loss_nornomalised_list_1=[]
loss_nornomalised_list_2=[]
loss_nornomalised_list_3=[]
loss_nornomalised_list_4=[]
loss_nornomalised_list_wrong=[]

for batch_idx, batch in enumerate(test_dataloader_selected):
    image, labels=batch[0].float(),batch[1].float()
        
    with torch.no_grad():
        if torch.cuda.is_available():
            image,labels=image.cuda(), labels.cuda()
            model.cuda()
        preds=model(image)

        grad_cam_hooks = {'forward': model.densenet121.features.norm5, 'backward': model.densenet121.classifier[0]}

        # cam = LayerGradCam(model, layer=model.densenet121.features)
        # cam=grad_cam(model, image, grad_cam_hooks,cls_idx=torch.LongTensor([0] * args.bs).cuda())
        attr_classes=[grad_cam(model, image, grad_cam_hooks,cls_idx=torch.LongTensor([i] * image.shape[0]).cuda()).cpu().numpy() for i in range(args.nc)]

        # for c, cam in enumerate(attr_classes):
        #     cam=cam[0].squeeze(0)
        #     cam=(cam-np.amin(cam))/(np.amax(cam)-np.amin(cam))
        #     plt.imshow(cam)
        #     saving_cam_path="/home/shelley/Documents/diffbygraph/cam/seed{}/".format(str(args.seed))
        #     os.makedirs(saving_cam_path, exist_ok=True)
        #     plt.savefig(saving_cam_path+"cam_{}".format(exist_labels[c]))
        #     plt.close()
        # attr_classes = [torch.Tensor(cam.attribute(image, [i] * image.shape[0])).detach().cpu().numpy() for i in range(args.nc)]
        # get the class map embedding by reinjecting to the model 

        classmap_embeddings=[]
        for i in range(len(attr_classes)):

            class_maps=attr_classes[i]
            class_maps=(class_maps-np.amin(class_maps))/(np.amax(class_maps)-np.amin(class_maps))

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

            if (torch.eq(Y_hat,patient_label).all().cpu().numpy()) and (count_correct1<20) and (patient_disease[0]==exist_labels[0]):
                pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
                loss_nornomalised_list_0=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                        patient_ind=count_correct1, patient_disease=patient_disease,preds=pred_disease, loss_nornomalised_list=loss_nornomalised_list_0)
                saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/correct_preded_single/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/{}/".format(args.name, args.interp, args.lr, args.epochs,patient_disease[0])
                os.makedirs(saving_graph_path,exist_ok=True)
                plt.savefig(saving_graph_path+"patient{}.png".format(count_correct1))
                plt.close()
                count_correct1+=1


            elif (torch.eq(Y_hat,patient_label).all().cpu().numpy()) and (count_correct2<20) and (patient_disease[0]==exist_labels[1]):
                pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
                loss_nornomalised_list_1=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                        patient_ind=count_correct2, patient_disease=patient_disease,preds=pred_disease, loss_nornomalised_list=loss_nornomalised_list_1)
                saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/correct_preded_single/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/{}/".format(args.name, args.interp, args.lr, args.epochs,patient_disease[0])
                os.makedirs(saving_graph_path,exist_ok=True)
                plt.savefig(saving_graph_path+"patient{}.png".format(count_correct2))
                plt.close()
                count_correct2+=1


            elif (torch.eq(Y_hat,patient_label).all().cpu().numpy()) and (count_correct3<20) and (patient_disease[0]==exist_labels[2]):
                pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
                loss_nornomalised_list_2=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                        patient_ind=count_correct3, patient_disease=patient_disease,preds=pred_disease, loss_nornomalised_list=loss_nornomalised_list_2)
                saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/correct_preded_single/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/{}/".format(args.name, args.interp, args.lr, args.epochs,patient_disease[0])
                os.makedirs(saving_graph_path,exist_ok=True)
                plt.savefig(saving_graph_path+"patient{}.png".format(count_correct3))
                plt.close()
                count_correct3+=1


            elif (torch.eq(Y_hat,patient_label).all().cpu().numpy()) and (count_correct4<20) and (patient_disease[0]==exist_labels[3]):
                pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
                loss_nornomalised_list_3=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                        patient_ind=count_correct4, patient_disease=patient_disease,preds=pred_disease, loss_nornomalised_list=loss_nornomalised_list_3)
                saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/correct_preded_single/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/{}/".format(args.name, args.interp, args.lr, args.epochs,patient_disease[0])
                os.makedirs(saving_graph_path,exist_ok=True)
                plt.savefig(saving_graph_path+"patient{}.png".format(count_correct4))
                plt.close()
                count_correct4+=1


            elif (torch.eq(Y_hat,patient_label).all().cpu().numpy()) and (count_correct5<20) and (patient_disease[0]==exist_labels[4]):
                pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
                loss_nornomalised_list_4=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                        patient_ind=count_correct5, patient_disease=patient_disease,preds=pred_disease, loss_nornomalised_list=loss_nornomalised_list_4)
                saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/correct_preded_single/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/{}/".format(args.name, args.interp, args.lr, args.epochs,patient_disease[0])
                os.makedirs(saving_graph_path,exist_ok=True)
                plt.savefig(saving_graph_path+"patient{}.png".format(count_correct5))
                plt.close()      
                count_correct5+=1


            # if (torch.eq(Y_hat,patient_label).all().cpu().numpy()) and count_wrong<20:
            #     pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
            #     loss_nornomalised_list_correct=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
            #                             patient_ind=count_correct, patient_disease=patient_disease,preds=pred_disease,loss_nornomalised_list=loss_nornomalised_list_correct)
            #     saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/correct_preded_multiple/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/".format(args.name, args.interp, args.lr, args.epochs)
            #     os.makedirs(saving_graph_path,exist_ok=True)
            #     plt.savefig(saving_graph_path+"patient{}.png".format(count_correct))
            #     plt.close()
            #     count_correct+=1


            elif (not torch.eq(Y_hat,patient_label).all().cpu().numpy()) and count_wrong<20:
                pred_disease=[exist_labels[i] for i, label in enumerate(Y_hat) if label==1]
                loss_nornomalised_list_wrong=plot_similarity_ingraph(torch.Tensor(patients_maps), classes_map_bypatient[p], image[p], compute_similarity,  exist_labels=exist_labels,
                                        patient_ind=count_wrong, patient_disease=patient_disease,preds=pred_disease, loss_nornomalised_list=loss_nornomalised_list_wrong)
                saving_graph_path="/home/shelley/Documents/diffbygraph/graph_visualisation_gradcamnew/seed{}/wrong_preded_multiple/".format(str(args.seed))+"{}_img_{}_lr{}_epoches{}/".format(args.name, args.interp, args.lr, args.epochs)
                os.makedirs(saving_graph_path,exist_ok=True)
                plt.savefig(saving_graph_path+"patient{}.png".format(count_wrong))
                plt.close()
                count_wrong+=1
  
# mutliple disease density plot
# plt.close()
# sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_correct)),4), label="correct", kde=True,log_scale=True) 
# sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_wrong)),4), label="wrong",kde=True,log_scale=True)
# plt.legend()
# plt.title("multiple disease correct and wrong similarity comparison")
# density_saving_path="/home/shelley/Documents/diffbygraph/densityplots/"
# os.makedirs(density_saving_path,exist_ok=True)
# plt.savefig(density_saving_path+"densityplot_multiple.png")

plt.close()
aa=np.concatenate(loss_nornomalised_list_0)
sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_0)),4), label=exist_labels[0]+str(round(roc_auc_byclass[0],3)), kde=True,log_scale=True)
sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_1)),4), label=exist_labels[1]+str(round(roc_auc_byclass[1],3)), kde=True,log_scale=True)
sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_2)),4), label=exist_labels[2]+str(round(roc_auc_byclass[2],3)), kde=True,log_scale=True)
sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_3)),4), label=exist_labels[3]+str(round(roc_auc_byclass[3],3)), kde=True,log_scale=True)
sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_4)),4), label=exist_labels[4]+str(round(roc_auc_byclass[4],3)), kde=True,log_scale=True)

sns.histplot(np.around(np.array(np.concatenate(loss_nornomalised_list_wrong)),4), label="wrong",kde=True,log_scale=True)
plt.legend()
plt.title("multiple disease correct and wrong similarity comparison")
density_saving_path="/home/shelley/Documents/diffbygraph/densityplots/"
os.makedirs(density_saving_path,exist_ok=True)
plt.savefig(density_saving_path+"densityplot_single.png")
            

            



 









