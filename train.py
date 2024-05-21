from umvi import *
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
import torch.utils.data as Data
from tqdm import tqdm
import os
from PIL import Image, ImageFile
import open_clip
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import numpy as np
import random
from utils.data import random_rotate_z, normalize_pc, augment_pc
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.autograd.profiler as profiler
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    size_in_kb = size / 1024
    size_in_mb = size_in_kb / 1024
    size_in_gb = size_in_mb / 1024
    
    return size_in_mb

class umvi_dataset(Dataset):
    def __init__(self,data_path_list):
        self.data_path_list=data_path_list

    def __getitem__(self,i):
        if ".npy" in self.data_path_list[i]:
            xyz, feat, image_feat,text_feat = load_npy_with_class(flie_path=self.data_path_list[i],num_points=10000,y_up=True,flag_objaverse=False)
            text_feat = random.choice(text_feat)
            text_feat = torch.from_numpy(text_feat).type(torch.float32)
        else:
            xyz, feat, image_feat,text_feat = load_npy_with_class(flie_path=self.data_path_list[i].replace("objaverse", "objaverse_pointcloud") + ".npy", num_points=10000,y_up=True, flag_objaverse=True)
            text_feat = random.choice(text_feat)
            text_feat = torch.from_numpy(text_feat).type(torch.float32)
        return image_feat,text_feat,xyz,feat

    def __len__(self):
        return len(self.data_path_list)

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


gpt4_filtering = json.load(open("./dataset/meta_data/gpt4_filtering.json", "r"))
def load_npy_with_class(flie_path,num_points=10000,y_up=True,flag_objaverse=True):
    data = np.load(flie_path, allow_pickle=True).item()
    n = data['xyz'].shape[0]
    idx = random.sample(range(n), num_points)
    xyz = data['xyz'][idx]
    rgb = data['rgb'][idx]

    text_feat=[]
    blip_caption_feat = data['blip_caption_feat']['original']
    msft_caption_feat = data['msft_caption_feat']['original']

    text_feat.append(blip_caption_feat)
    text_feat.append(msft_caption_feat)

    if flag_objaverse == True:
        uid= data["id"]
        if not (gpt4_filtering[uid]["flag"] == "N"):
            text_feat.append(data["text_feat"][0]["prompt_avg"])
        pass
    else:
        text_feat.append(data["text_feat"][0]["prompt_avg"])


    if len(data["retrieval_text"]) > 0:
        idx = np.random.randint(len(data["retrieval_text"]))
        text_feat.append(data["retrieval_text_feat"][idx]["original"])

    image_feat = data['image_feat']
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]

    xyz = random_rotate_z(augment_pc(normalize_pc(xyz)))

    if random.random()>0.5:
        rgb = torch.tensor([0.4, 0.4, 0.4]).unsqueeze(0).repeat(10000, 1)

    features = np.concatenate([xyz, rgb], axis=1)

    xyz = torch.from_numpy(xyz).type(torch.float32)
    assert not np.isnan(xyz).any()

    return xyz,torch.from_numpy(features).type(torch.float32),torch.from_numpy(image_feat).type(torch.float32), text_feat



class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input1, input2):
        cosine_similarity = F.cosine_similarity(input1, input2, dim=1)
        
        loss = 1 - cosine_similarity.mean()
        
        return loss


def retrieve_score(image_features_tensors,text_features_tensors):
    num_samples = image_features_tensors.shape[0]
    #image ->text
    similarity_matrix = torch.matmul(image_features_tensors, text_features_tensors.t())

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
    similarity_matrix = torch.matmul(text_features_tensors, image_features_tensors.t())

    _, indices = similarity_matrix.topk(10, dim=1, largest=True, sorted=True)

    indices = indices.to("cpu")

    r_at_1 = (indices[:, 0] == torch.arange(num_samples)).float().mean().item()
    r_at_5 = (indices[:, :5] == torch.tensor([[i] * 5 for i in range(num_samples)])).sum(dim=1).clamp_max(1).float().mean().item()
    r_at_10 = (indices[:, :10] == torch.tensor([[i] * 10 for i in range(num_samples)])).sum(dim=1).clamp_max(1).float().mean().item()
    print("----------text retrieve 3D model -------------")
    print(f'R@1: {r_at_1}')
    print(f'R@5: {r_at_5}')
    print(f'R@10: {r_at_10}')


def extract_crossmodal_feature(caption_dict,path_3d,N_sample,device):
    xyz, feat,blip_text_feat,ms_text_feat,images_feat= load_npy(path_3d.replace("objaverse", "objaverse_pointcloud") + ".npy")
    xyz = torch.unsqueeze(xyz,0)
    feat = torch.unsqueeze(feat,0)
    images_feat = images_feat.unsqueeze(0)
    with torch.no_grad():
        three_d_features = model.module(images_feat.to(device),N_sample,xyz.to(device),feat.to(device),None,False,None)
    return three_d_features




def evaluate_on_manual(model,N_sample,device):
    model.module.eval()
    num_samples=1000
    with open('./dataset/objaverse_test_text.json', 'r') as f:
        caption_dict=json.load(f)
    test_path_list=list(caption_dict.keys())
    image_features_list=[]
    text_features_list=[]
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            image_features = extract_crossmodal_feature(caption_dict,test_path_list[i],N_sample,device)
            image_features_list.append(image_features)
        image_features_tensors=torch.squeeze(torch.stack(image_features_list))
        image_features_tensors /= image_features_tensors.norm(dim=-1, keepdim=True)
        text_features_tensors = torch.load("./text_manual_1280.pt",map_location=device)
        text_features_tensors /= text_features_tensors.norm(dim=-1, keepdim=True)
        retrieve_score(image_features_tensors,text_features_tensors)
    model.train()

torch.distributed.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

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
        self.learning_rate = 1e-8
        self.froze_visual_encoder=True
        self.batch_size=64

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


if local_rank == 0:
    attributes = dir(config)
    for attr in attributes:
        if not callable(getattr(config, attr)) and not attr.startswith("__"):
            print(attr, "=", getattr(config, attr))
    print("GPU:",device) 

with open("./data_path/train_all.json","r") as file:
    path_list=json.load(file)


number = torch.arange(len(path_list))
torch_dataset = umvi_dataset(path_list)
train_sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset,shuffle=True)
len_dataset = len(path_list)


loader = Data.DataLoader(
    dataset=torch_dataset,     
    batch_size=batch_size,     
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
)

if len_dataset % batch_size == 0:
    total_steps = (len_dataset // batch_size) * num_epochs
else:
    total_steps = (len_dataset // batch_size + 1) * num_epochs

model = Umvi_23(num_layers, d_model, nhead,device).to(device)
model.ngpu = torch.cuda.device_count()
logit_scale = LogitScaleNetwork(logit_scale_init).to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    logit_scale = torch.nn.parallel.DistributedDataParallel(logit_scale, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)


no_decay = ["a_attention","b_attention","c_attention","d_attention","f_attention","e_attention"]

special_layers_params_no_decay = [param for name, param in model.module.named_parameters() if any(layer_name in name for layer_name in ["point_model"]) and any(nd in name for nd in no_decay)]
special_layers_params_decay = [param for name, param in model.module.named_parameters() if any(layer_name in name for layer_name in ["point_model"]) and not any(nd in name for nd in no_decay)]
other_params_no_decay = [param for name, param in model.module.named_parameters() if not any(name.startswith(layer_name) for layer_name in ["point_model"]) and any(nd in name for nd in no_decay)]
other_params_decay = [param for name, param in model.module.named_parameters() if not any(name.startswith(layer_name) for layer_name in ["point_model"]) and not any(nd in name for nd in no_decay)]

optimizer_grouped_parameters = [
    {'params': special_layers_params_no_decay, 'lr': 1e-8, 'weight_decay': 0.0},
    {'params': special_layers_params_decay, 'lr': 1e-8, 'weight_decay': 0.03},
    {'params': other_params_no_decay, 'lr': 1e-8, 'weight_decay': 0.0},
    {'params': other_params_decay, 'lr': 1e-8, 'weight_decay': 0.03},
    {'params': [p for n, p in logit_scale.module.named_parameters()], 'lr': 1e-5, 'weight_decay': 0.0},
]


optimizer = optim.AdamW(optimizer_grouped_parameters)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10*total_steps, T_mult=1, eta_min=1e-9)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

for param in model.module.point_model.parameters():
    param.requires_grad = True


mse_loss = nn.MSELoss()
cosine_loss = CosineSimilarityLoss()

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(tqdm(loader)):
        images_feat,text_feat,xyz,feat = batch
        text_feat = text_feat.to(device)
        text_feat = text_feat.squeeze(1)
        optimizer.zero_grad()
        images_feat /= images_feat.norm(dim=-1, keepdim=True)
        if epoch<5:
            three_d_features = model(images_feat[:,0:10,:].to(device), N_sample, xyz.to(device), feat.to(device),text_feat,True,N_sample-epoch)
        else:
            three_d_features = model(images_feat[:,0:10,:].to(device), N_sample, xyz.to(device), feat.to(device),text_feat,True,5)

        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        loss,_ = model.module.contrastive_loss(three_d_features, text_feat, logit_scale=logit_scale.module.logit_scale)

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if (local_rank == 0) and (step % 1000 == 0) and (step !=0 ):
            print("eval on manual:")
            torch.cuda.empty_cache()
            evaluate_on_manual(model,N_sample,device) 
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')

    if local_rank == 0:
        print("eval on manual:")
        torch.cuda.empty_cache()
        evaluate_on_manual(model,N_sample,device)
        torch.save(model.module.state_dict(), "./model/model_pt/view+pointcloud_head_"+str(nhead)+"layer_"+str(num_layers)+"epoch_"+str(epoch)+".pt")