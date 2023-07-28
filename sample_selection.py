import os
import pandas as pd
import random

homepath="/home/shelley/Documents/"
traindata=homepath+"CheXpert/CheXpert-v1.0/train"
validdata=homepath+"CheXpert/CheXpert-v1.0/valid"
testdata=homepath+"CheXpert/CheXpert-v1.0/test"

train_labels=pd.read_csv(homepath+"CheXpert/CheXpert-v1.0/train.csv")
valid_labels=pd.read_csv(homepath+"CheXpert/CheXpert-v1.0/valid.csv")
test_labels=pd.read_csv(homepath+"CheXpert/CheXpert-v1.0/test_labels.csv")
exist_labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"] 


train_labels_front_withalllabel=pd.DataFrame([x for i, x in train_labels.iterrows() if "front" in x['Path']])
train_labels_atleastone_positive= train_labels_front_withalllabel[(train_labels_front_withalllabel[exist_labels]>0).any(axis=1)]# remove sample without positive case for selected labels
train_labels_atleastone_positive.loc[:,exist_labels]=train_labels_atleastone_positive[exist_labels].fillna(0)# not mention in report we assume it is a negative sample
train_labels_front=train_labels_atleastone_positive[(train_labels_atleastone_positive[exist_labels]>=0).all(axis=1)]# remove the rest uncertain
print("training front images:",train_labels_front.shape[0])

# sampling for subset for training
Al_ind=train_labels_front[train_labels_front[exist_labels[0]]>0].index
Ca_ind=train_labels_front[train_labels_front[exist_labels[1]]>0].index
Co_ind=train_labels_front[train_labels_front[exist_labels[2]]>0].index
Ed_ind=train_labels_front[train_labels_front[exist_labels[3]]>0].index
Pl_ind=train_labels_front[train_labels_front[exist_labels[4]]>0].index    

training_selected_ind=random.sample(list(Co_ind.values),9000)
selected_ca_sample=random.sample(set(Ca_ind).difference(set(training_selected_ind)), 9000)
training_selected_ind+=selected_ca_sample

selected_al_sample=random.sample(list(set(Al_ind).difference(set(training_selected_ind))), 9000)
training_selected_ind+=selected_al_sample

selected_ed_sample=random.sample(list(set(Ed_ind).difference(set(training_selected_ind))),9000)
training_selected_ind+=selected_ed_sample

selected_pl_sample=random.sample(list(set(Pl_ind).difference(set(training_selected_ind))), 9000)
training_selected_ind+=selected_pl_sample

selected_train_dataset=train_labels.iloc[sorted(training_selected_ind),:]
selected_train_dataset.loc[:,exist_labels]=selected_train_dataset[exist_labels].fillna(0)# not mention in report we assume it is a negative sample

valid_labels_front=pd.DataFrame([x for i, x in valid_labels.iterrows() if "front" in x['Path']])
test_labels_front=pd.DataFrame([x for i, x in test_labels.iterrows() if "front" in x['Path']])

print("avaliable training data",selected_train_dataset.shape[0])
print("validation data",valid_labels_front.shape[0])     
print("test data",test_labels_front.shape[0])
  
# PATH="/home/shelley/Documents/CheXpert/CheXpert-v1.0/withtestset_alltrain"
# os.makedirs(PATH, exist_ok=True)
# train_labels_front.to_csv(PATH+"/train.csv")
# valid_labels_front.to_csv(PATH+"/val.csv")
# test_labels_front.to_csv(PATH+"/test.csv")
