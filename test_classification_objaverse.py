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

device = "cuda:2" if torch.cuda.is_available() else "cpu"

def load_npy_batch(file_paths, num_points=10000, y_up=True):
    batch_xyz = []
    batch_features = []
    batch_image_feature= []
    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        if n > num_points:
            idx = random.sample(range(n), num_points)
        else:
            idx = np.arange(n)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]

        xyz = normalize_pc(xyz)
        features = np.concatenate([xyz, rgb], axis=1)
        image_feature =torch.from_numpy(data['image_feat']).float()
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        batch_xyz.append(torch.from_numpy(xyz).float())
        batch_features.append(torch.from_numpy(features).float())
        batch_image_feature.append(image_feature)

    batch_xyz = torch.stack(batch_xyz, dim=0).to(device)
    batch_features = torch.stack(batch_features, dim=0).to(device)
    batch_image_feature = torch.stack(batch_image_feature, dim=0).to(device)
    return batch_xyz, batch_features,batch_image_feature



def batched_model_inference(model, xyz_batch, feat_batch, img_feat_batch, config):
    batch_features = []
    for i in range(0, xyz_batch.size(0), batch_size):
        xyz_sub = xyz_batch[i:i + batch_size]
        feat_sub = feat_batch[i:i + batch_size]
        images_feat = img_feat_batch[i:i+batch_size]
        features = model(images_feat[:,0:N_sample,:].to(device),N_sample,xyz_sub, feat_sub,None,False,None)
        batch_features.append(features)
    return torch.cat(batch_features, dim=0)



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

def method_batch(three_d_features, categories, text_features, category_to_index):
    # Compute similarities between batch of image features and all text features
    with torch.no_grad():
        similarity = 100.0 * three_d_features @ text_features.T  # Shape: [batch_size, num_categories]

        # Get top-5 indices for each item in the batch
        top_k_values, top_k_indices = similarity.topk(5, dim=1, largest=True, sorted=True)

        # Convert categories to indices to compare
        category_indices = torch.tensor([category_to_index[cat] for cat in categories], device=device)
        category_indices = category_indices.unsqueeze(1)  # Shape: [batch_size, 1] for broadcasting

        # Calculate accuracies
        top1_correct = (top_k_indices[:, 0:1] == category_indices).float().sum()
        top3_correct = (top_k_indices[:, :3] == category_indices.expand(-1, 3)).any(dim=1).float().sum()
        top5_correct = (top_k_indices == category_indices.expand(-1, 5)).any(dim=1).float().sum()

    # Convert sums to proportions
    num_items = three_d_features.size(0)
    top1_accuracy = top1_correct 
    top3_accuracy = top3_correct 
    top5_accuracy = top5_correct 

    return top1_accuracy, top3_accuracy, top5_accuracy




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
        self.batch_size=30

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

with open('./dataset/meta_data/split/lvis.json', 'r') as f:
    data_name_dict = json.load(f)

data = list(data_name_dict)
categories = sorted(np.unique([data_item['category'] for data_item in data]))
category2idx = {categories[i]: i for i in range(len(categories))}
num_samples = len(data)
print(num_samples)
batch_size = 100  
text_feature = torch.tensor(np.load("./dataset/meta_data/lvis_cat_name_pt_feat.npy")).to(device)
text_feature = F.normalize(text_feature, dim=-1)
results = {'top1': 0, 'top3': 0, 'top5': 0}
with torch.no_grad():
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_paths = [data[j]["data_path"] for j in range(i, min(i + batch_size, num_samples))]
        batch_categories = [data[j]["category"] for j in range(i, min(i + batch_size, num_samples))]
        
        xyz_batch, feat_batch, img_feat_batch = load_npy_batch(batch_paths)
        three_d_features = batched_model_inference(model, xyz_batch, feat_batch, img_feat_batch, config)
        three_d_features = F.normalize(three_d_features, dim=-1)
        
        top1_accuracy, top3_accuracy, top5_accuracy = method_batch(three_d_features, batch_categories, text_feature, category2idx)
        
        results['top1'] += top1_accuracy
        results['top3'] += top3_accuracy
        results['top5'] += top5_accuracy
        torch.cuda.empty_cache()

print(results['top1']/num_samples, results['top3']/num_samples, results['top5']/num_samples)

