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

device = "cuda:7" if torch.cuda.is_available() else "cpu"

def load_npy(flie_path,num_points=10000,y_up=True):
    data = np.load(flie_path, allow_pickle=True).item()
    n = data['xyz'].shape[0]
    idx = random.sample(range(n), num_points)
    xyz = data['xyz'][idx]
    rgb = data['rgb'][idx]

    blip_caption_feat = data['blip_caption_feat']['original']
    msft_caption_feat = data['msft_caption_feat']['original']
    image_feat = data['image_feat']
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]

    xyz = normalize_pc(xyz)

    features = np.concatenate([xyz, rgb], axis=1)

    xyz = torch.from_numpy(xyz).type(torch.float32)
    assert not np.isnan(xyz).any()

    return xyz,torch.from_numpy(features).type(torch.float32),torch.from_numpy(blip_caption_feat).type(torch.float32),torch.from_numpy(msft_caption_feat).type(torch.float32),torch.from_numpy(image_feat).type(torch.float32)


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
        random_numbers = random.sample(range(12), N_sample)
        features = model(images_feat[:,random_numbers,:].to(device),N_sample,xyz_sub, feat_sub)
        batch_features.append(features)
    return torch.cat(batch_features, dim=0)


def retrieve_score(three_d_features_tensors,text_features_tensors):
    num_samples = three_d_features_tensors.shape[0]
    #image ->text
    similarity_matrix = torch.matmul(three_d_features_tensors, text_features_tensors.t())

    _, indices = similarity_matrix.topk(10, dim=1, largest=True, sorted=True)

    indices = indices.to("cpu")

    r_at_1 = (indices[:, 0] == torch.arange(num_samples)).float().mean().item()
    r_at_5 = (indices[:, :5] == torch.tensor([[i] * 5 for i in range(num_samples)])).sum(dim=1).clamp_max(1).float().mean().item()
    r_at_10 = (indices[:, :10] == torch.tensor([[i] * 10 for i in range(num_samples)])).sum(dim=1).clamp_max(1).float().mean().item()

    print("----------3D model retrieve text -------------")
    print(f'R@1: {r_at_1}')
    print(f'R@5: {r_at_5}')
    print(f'R@10: {r_at_10}')

    #text ->image
    similarity_matrix = torch.matmul(text_features_tensors, three_d_features_tensors.t())

    _, indices = similarity_matrix.topk(10, dim=1, largest=True, sorted=True)

    indices = indices.to("cpu")

    r_at_1 = (indices[:, 0] == torch.arange(num_samples)).float().mean().item()
    r_at_5 = (indices[:, :5] == torch.tensor([[i] * 5 for i in range(num_samples)])).sum(dim=1).clamp_max(1).float().mean().item()
    r_at_10 = (indices[:, :10] == torch.tensor([[i] * 10 for i in range(num_samples)])).sum(dim=1).clamp_max(1).float().mean().item()
    print("----------text retrieve 3D model -------------")
    print(f'R@1: {r_at_1}')
    print(f'R@5: {r_at_5}')
    print(f'R@10: {r_at_10}')

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
model = Umvi_23_2(num_layers, d_model, nhead,device).to(device)

state_dict = torch.load("./model/model_pt/final.pt", map_location=torch.device('cpu'))
if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

num_samples=1000

with open('./dataset/objaverse_test_text.json', 'r') as f:
    caption_dict=json.load(f)

test_path_list=list(caption_dict.keys())
print(len(test_path_list))


three_d_features_list=[]
text_features_tensors = torch.load("./text_manual_text_objaverse_1280.pt",map_location=device)
text_features_tensors /= text_features_tensors.norm(dim=-1, keepdim=True)

with torch.no_grad():
    for i in tqdm(range(num_samples)):
        xyz, feat,blip_text_feat,ms_text_feat,images_feat = load_npy(test_path_list[i].replace("objaverse", "objaverse_pointcloud") + ".npy")
        xyz=torch.unsqueeze(xyz.to(device), 0)
        feat=torch.unsqueeze(feat.to(device), 0)
        images_feat=torch.unsqueeze(images_feat.to(device), 0)
        three_d_features = model(images_feat[:,0:10,:].to(device),N_sample,xyz, feat)
        three_d_features = F.normalize(three_d_features, dim=1)
        three_d_features_list.append(torch.squeeze(three_d_features))
        torch.cuda.empty_cache()
    three_d_features_tensors=torch.stack(three_d_features_list)
    
    retrieve_score(three_d_features_tensors,text_features_tensors)

