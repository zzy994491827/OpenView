import torch
import clip
from PIL import Image
import os
import pickle
import random
import open_clip
import json
import numpy as np
import open3d as o3d
import random
import torch
import sys
from param import parse_args
import MinkowskiEngine as ME
from utils.data import normalize_pc
from utils.misc import load_config
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import open_clip
import re
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from umvi import *

random.seed(42)

# Seed NumPy's random module
np.random.seed(42)

# Seed PyTorch (both CPU and GPU if available)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def method(three_d_features,category,text_features):
    #scores of average image features with text features
    #the result is best
    true_top1=False
    true_top3=False
    true_top5=False
    with torch.no_grad():
        similarity = (100.0 * three_d_features @ text_features.T)
        _,index_x=similarity.topk(5, dim=1, largest=True, sorted=True)
        index=index_x[0]
        if categories[index[0]] == category:
            true_top1=True
            true_top3=True
            true_top5=True
        elif (categories[index[1]] == category) or (categories[index[2]] == category):
            true_top3=True
            true_top5=True
        elif (categories[index[3]] == category) or (categories[index[4]] == category):
            true_top5=True
        
        return true_top1,true_top3,true_top5




class config():
    def __init__(self):
        self.d_model = 1280
        self.nhead = 32
        self.num_layers = 6
        self.N_sample = 10
        self.logit_scale_init=14.28
        self.num_epochs = 15
        self.warm_up_ratio = 0.1
        self.loss_name ="contra"
        self.learning_rate = 1e-6
        self.froze_visual_encoder=True
        self.batch_size=1

config=config()

d_model = config.d_model
nhead = config.nhead
num_layers = config.num_layers
N_sample = config.N_sample
logit_scale_init=config.logit_scale_init
num_epochs = config.num_epochs
warm_up_ratio=config.warm_up_ratio
loss_name=config.loss_name
learning_rate=config.learning_rate
froze_visual_encoder=config.froze_visual_encoder
batch_size=config.batch_size

print("Loading model...")
model = Umvi_23(num_layers, d_model, nhead,device).to(device)

state_dict = torch.load("./model/model_pt/final.pt", map_location=torch.device('cpu'))
if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

with open('./dataset/meta_data/modelnet40/test_split.json', 'r') as f:
    data_name_dict = json.load(f)

name_feat = np.load("./dataset/meta_data/modelnet40/cat_name_pt_feat.npy",allow_pickle=True).item()
pc_list = np.load("./dataset/meta_data/modelnet40/test_pc_with_image.npy",allow_pickle=True)

data = list(data_name_dict)
categories = list(name_feat.keys())
text_features_list=[]
for i in range(len(categories)):
    text_features_list.append(torch.from_numpy(name_feat[categories[i]]["open_clip_text_feat"]).type(torch.float32))
text_features = torch.stack(text_features_list).squeeze()
text_features = F.normalize(text_features, dim=-1)
num_samples = len(data)
print(num_samples)
results = {'top1': 0, 'top3': 0, 'top5': 0}
with torch.no_grad():
    for i in tqdm(range(0, num_samples)):
        category=data[i]["category"]
        pc=pc_list[i]
        n = pc['xyz'].shape[0]
        idx = random.sample(range(n), 10000)
        xyz = pc['xyz'][idx]
        rgb = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(0).repeat(10000, 1)

        images_feat = pc['image_feat']
        images_feat /= images_feat.norm(dim=-1, keepdim=True)
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
        xyz = normalize_pc(xyz)
        pc_features =  torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).type(torch.float32)
        xyz = torch.from_numpy(xyz).type(torch.float32)
        
        xyz=xyz.unsqueeze(0)
        pc_features=pc_features.unsqueeze(0)
        images_feat=images_feat.unsqueeze(0)

        three_d_features = model(images_feat[:,0:10,:].to(device), N_sample, xyz.to(device), pc_features.to(device),None,False,None)
        three_d_features = F.normalize(three_d_features, dim=-1)
        top1_accuracy, top3_accuracy, top5_accuracy = method(three_d_features,category,text_features.to(device))

        results['top1'] += top1_accuracy
        results['top3'] += top3_accuracy
        results['top5'] += top5_accuracy
        torch.cuda.empty_cache()

print(results['top1']/num_samples, results['top3']/num_samples, results['top5']/num_samples)

