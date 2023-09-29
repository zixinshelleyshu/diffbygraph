import os
import numpy as np
import torch
import networkx as nx
import cv2
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
import pandas as pd



#calculating similarity
class compute_similarity(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=1)
        self.device = device

    def view_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1)

    def forward(self, sal_tensor_list: torch.Tensor) -> torch.Tensor:
        loss_list = torch.Tensor([]).to(self.device)
        for sal_comb in itertools.combinations(sal_tensor_list, 2):
            loss_list = torch.cat((loss_list, torch.unsqueeze(torch.abs(self.cossim(self.view_tensor(sal_comb[0]), self.view_tensor(sal_comb[1]))).mean(), dim=0)))
        return torch.mean(loss_list)
    
    def similarity_scores(self, sal_tensor_list: torch.Tensor) -> pd.DataFrame:
        loss_list=[]
        indexes=range(sal_tensor_list.shape[0])
        seleced_index=[index_comb for index_comb in itertools.combinations(indexes,2)]

        for sal_comb in itertools.combinations(sal_tensor_list, 2):
            # dissmilarity depends on conssim 
            # heat_map=torch.unsqueeze(torch.abs(self.cossim(self.view_tensor(sal_comb[0]), self.view_tensor(sal_comb[1])).mean()), dim=0).numpy()
            # loss_list.append(list(heat_map))
            # dissimilarity depends on abs 
            loss_list= np.concatenate((loss_list,torch.unsqueeze(torch.abs(self.view_tensor(sal_comb[0])-self.view_tensor(sal_comb[1])).mean(), dim=0).numpy()))
        # loss_list=np.concatenate(loss_list)
        max_dis=np.max(loss_list)
        loss_list_nornomalised=loss_list
        loss_list=loss_list/np.max(loss_list)
        similarity_df=pd.DataFrame(zip(seleced_index,loss_list))
        return similarity_df, max_dis, loss_list_nornomalised
        

def plot_similarity_ingraph(images, embeddings,orig_image, ClassDistinctivenessLoss, exist_labels,patient_ind, patient_disease,preds,loss_nornomalised_list):
    cdcriterion = ClassDistinctivenessLoss(device="cuda")
    similarity_df, max_dis, loss_nornomalised= cdcriterion.similarity_scores(torch.tensor(embeddings))
    loss_nornomalised_list.append(loss_nornomalised)
    
    
    # Generate the computer network graph
    G = nx.Graph()
    # G=Network()
    for i in range(images.shape[0]):
        if i==0:
            resizing=Resize(size=(orig_image.shape[1], orig_image.shape[2]), antialias=True)
        heatmap=resizing(images[i])
        if i==0:
            orig_image=orig_image.permute((1,2,0)).detach().cpu().numpy()
            orig_image=(orig_image-np.amin(orig_image))/(np.amax(orig_image)-np.amin(orig_image))

            # gray_image = cv2.cvtColor(orig_image.detach().cpu().numpy(), cv2.COLOR_BGR2GRAY)
        heatmap=np.squeeze(heatmap.detach().cpu().numpy(), axis=0)
        # heatmapshow=None
        # heatmap = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # heatmapshow=cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)
        heatmap=(heatmap-np.amin(heatmap))/(np.amax(heatmap)-np.amin(heatmap))
        heatmap= cv2.applyColorMap(np.uint8((255-heatmap)*255), cv2.COLORMAP_JET)
        # heatmap_3channels = cv2.merge((heatmap,heatmap,heatmap))

        
        # orig_image=PIL.Image.fromarray(np.uint8(orig_image*255))
        # orig_image=orig_image.convert('L') 
        # heat_images = cv2.addWeighted(np.uint8(255*heatmap), 0.5, orig_image, 0.5, 0)

        heat_images=np.uint8(orig_image*0.7*255)+heatmap*0.3
        heat_images=(heat_images-np.amin(heat_images))/(np.amax(heat_images)-np.amin(heat_images))
        # G.add_node(i,label=exist_labels[i][:2])
        G.add_node(i,label=exist_labels[i][:2], image=heat_images)


     # G.add_edges_from(similarity_df[0],length=similarity_df[1])
    for i in range(similarity_df.shape[0]):
        G.add_edge(similarity_df[0][i][0],similarity_df[0][i][1],weight=similarity_df[1][i])
        # G.add_edges_from([similarity_df[0][i]],weight=similarity_df[1][i])


    labels = {}
    labels[0]=exist_labels[0][:2]
    labels[1]=exist_labels[1][:2]
    labels[2]=exist_labels[2][:2]
    labels[3]=exist_labels[3][:2]
    labels[4]=exist_labels[4][:2]

    # G.save_graph("test.html")
    nx.draw(G,with_labels=True)

    # Get a reproducible layout and create figure
    pos = nx.spring_layout(G, seed=1734289230)
    fig, ax = plt.subplots()


    # Note: the min_source/target_margin kwargs only work with FancyArrowPatch objects.
    # Force the use of FancyArrowPatch for edge drawing by setting `arrows=True`,
    # but suppress arrowheads with `arrowstyle="-"`
    nx.draw_networkx_edges(
    G,
    pos=pos,
    ax=ax,
    arrows=True,
    arrowstyle="-",
    min_source_margin=15,
    min_target_margin=15
    )

# Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
# Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

# Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
    icon_center = icon_size / 2.0

    pos_higher = {}
    y_off = 0.18  # offset on the y axis

    for k, v in pos.items():
        # pos_higher[k] = (v[0],v[1]+y_off)
        pos_higher[k] = (v[0]+y_off,v[1])

    nx.draw_networkx_labels(G, pos_higher, labels,font_size=10)

    ax.set_title("patient {}th has{}, pred{}".format(patient_ind, patient_disease, preds),fontsize=8)
    ax.text(ax.get_xlim()[0],ax.get_ylim()[0], "max dissimilarity:{}".format(round(max_dis, 6)))
# Add the respective image to each node
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
    # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        a.axis("off")

    return loss_nornomalised_list 