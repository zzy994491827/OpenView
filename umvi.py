import open_clip
import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
import os
from param import parse_args
import sys
import open3d as o3d
import MinkowskiEngine as ME
from utils.data import normalize_pc
from utils.misc import load_config
import Minkowski, ppat
from collections import OrderedDict
import re
import random



class LogitScaleNetwork(nn.Module):
    def __init__(self, init_scale=1 / 0.07):
        super(LogitScaleNetwork, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale))

def make(config):
    if config.model.name == "MinkowskiFCNN":
        model = Minkowski.MinkowskiFCNN(config)
    elif config.model.name == "MinkResNet":
        model = Minkowski.MinkResNet(config)
    elif config.model.name == "MinkResNet34":
        model = Minkowski.MinkResNet34(config)
    elif config.model.name == "MinkResNet11":
        model = Minkowski.MinkResNet11(config)
    elif config.model.name == "MinkowskiFCNN_small":
        model = Minkowski.MinkowskiFCNN_small(config)
    elif config.model.name == "PointBERT":
        model = ppat.make(config)
    elif config.model.name == "DGCNN":
        from . import dgcnn
        model = dgcnn.make(config)
    elif config.model.name == "PointNeXt":
        from . import pointnext
        model = pointnext.make(config)
    elif config.model.name == "PointMLP":
        from . import pointmlp
        model = pointmlp.make(config)
    elif config.model.name == "PointNet":
        from . import pointnet
        model = pointnet.make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model

def remove_top_tokens(a, B,number_to_keep):
    cos_sim = F.cosine_similarity(a.unsqueeze(1), B, dim=2)
    
    sorted_sim, sorted_indices = torch.sort(cos_sim, dim=1, descending=False)
    
    num_tokens_to_keep = number_to_keep
    
    B_filtered = []
    
    for i in range(B.size(0)):
        indices_to_keep = sorted_indices[i, 0:num_tokens_to_keep]
        
        B_filtered.append(B[i, indices_to_keep])
    
    
    return torch.stack(B_filtered)



class FusionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(FusionLayer, self).__init__()
        self.self_attn_image = Attention(d_model, nhead)
        self.self_attn_point = Attention(d_model, nhead)
        self.self_attn_fusion = Attention(d_model, nhead)
        self.self_attn_fusion_x = Attention(d_model, nhead)
        self.self_attn_fusion_y = Attention(d_model, nhead)
        self.a_attention = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.b_attention = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.c_attention = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.d_attention = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.feed_forward_image = nn.Sequential(
            nn.Linear(d_model, 4096),
            nn.ReLU(),
            nn.Linear(4096, d_model)
        )
        self.feed_forward_point = nn.Sequential(
            nn.Linear(d_model, 4096),
            nn.ReLU(),
            nn.Linear(4096, d_model)
        )
        self.feed_forward_fusion = nn.Sequential(
            nn.Linear(d_model, 4096),
            nn.ReLU(),
            nn.Linear(4096, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_image = nn.LayerNorm(d_model)
        self.norm2_point = nn.LayerNorm(d_model)
        self.norm2_fusion = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self,x,y):
        attn_output_image, _ = self.self_attn_image(x, x, x,None)
        x1 = self.norm1(x + attn_output_image)
        ff_output_image = self.feed_forward_image(x1)
        x1 = self.norm2_image(x1 + ff_output_image)

        attn_output_piont, _ = self.self_attn_point(y, y, y,None)
        y1 = self.norm1(y + attn_output_piont)
        ff_output_point = self.feed_forward_point(y1)
        y1 = self.norm2_point(y1 + ff_output_point)

        attn_output_fusion_x, _ = self.self_attn_fusion_x(y, y, x,None)
        attn_output_fusion_y, _ = self.self_attn_fusion_y(x, x, y,None)

        f = torch.cat((x,y),dim=1)
        weight_1=torch.clamp(self.a_attention, min=0, max=1)
        weight_2=torch.clamp(self.b_attention, min=0, max=1)

        fusion_x=weight_1*attn_output_fusion_x+(1-weight_1)*attn_output_image
        fusion_y=weight_2*attn_output_fusion_y+(1-weight_2)*attn_output_piont

        f1 = self.norm1(f+torch.cat((fusion_x,fusion_y),dim=1))

        ff_output_fusion = self.feed_forward_fusion(f1)
        f2 = self.norm2_fusion(f1 + ff_output_fusion)


        x2=f2[:,0:x.shape[1],:]
        y2=f2[:,x.shape[1]:,:]

        weight_3=torch.clamp(self.c_attention, min=0, max=1)
        weight_4=torch.clamp(self.d_attention, min=0, max=1)

        x_output=self.norm3(weight_3*x1+(1-weight_3)*x2)
        y_output=self.norm3(weight_4*y1+(1-weight_4)*y2)
        return x_output, y_output

#full+hard
class Umvi_23_2(nn.Module):
    def __init__(self, num_layers, d_model, nhead, device):
        super(Umvi_23_2, self).__init__()
        self.fusion_layers = nn.ModuleList([FusionLayer(d_model, nhead) for _ in range(num_layers)])
        self.d_model=d_model
        self.device=device
        self.ngpu=1
        cli_args, extras = parse_args(sys.argv[1:])
        self.config = load_config("./configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
        self.point_model = self.load_model(self.config)
        self.point_embedding = nn.Parameter(torch.randn(1,d_model), requires_grad=True)
        self.image_embedding = nn.Parameter(torch.randn(1,d_model), requires_grad=True)
        self.f_attention = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.e_attention = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    def forward(self, image_features, N_sample, xyz, feat, text_feat,train,number_image_use):
        point_features = F.normalize(self.point_model(xyz, feat, device=self.device, quantization_size=self.config.model.voxel_size), dim=-1)
        point_features_last = torch.cat((point_features[:,0,:].unsqueeze(1),point_features[:,1:20,:] + self.point_embedding), dim=1)
        
        if train:
            image_features_remove=remove_top_tokens(text_feat,image_features,number_image_use)
        else:
            image_features_remove=image_features
        average_features = F.normalize(torch.mean(image_features_remove, dim=1),dim=-1)

        image_features = image_features_remove + self.image_embedding.unsqueeze(1)
        average_features = average_features.unsqueeze(1)
        image_d_features_last = torch.cat((average_features,image_features), dim=1)

        for layer in self.fusion_layers:
            image_d_features_new, point_features_new = layer(image_d_features_last, point_features_last)
            image_d_features_last = image_d_features_new + image_d_features_last
            point_features_last = point_features_new + point_features_last
            return F.normalize(F.normalize(average_features.squeeze() + point_features[:,0,:],dim=-1) + F.normalize(image_d_features_last[:,0,:] + point_features_last[:,0,:],dim=-1),dim=-1)
    def contrastive_loss(self,feat1, feat2, logit_scale=1, mask = None):
        if self.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss,accuracy

    def load_model(self, config, model_name="OpenShape/openshape-spconv-all"):
        config.model.scaling = 7
        model = make(config).to(self.device)

        if config.model.name.startswith('Mink'):
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        checkpoint = torch.load("/home/zhouziyang/pyproject/UMVI/openshape/pointbert-no-lvis.pt",map_location=self.device)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in checkpoint['state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
        model.load_state_dict(model_dict)
        return model



class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
       
        assert d_model % self.num_heads == 0
       
        self.depth = d_model // self.num_heads
       
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
       
        self.dense = nn.Linear(d_model, d_model)
       
    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
   
    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
       
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
       
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q_x, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k_x, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v_x, depth)
       
        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
       
        # Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Add the mask to the scaled tensor.
           
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (batch_size, num_heads, seq_len_q_x, seq_len_k_x)
       
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q_x, depth)
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q_x, num_heads, depth)
        output = output.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len_q_x, d_model)
       
        output = self.dense(output)  # (batch_size, seq_len_q_x, d_model)
       
        return output, attention_weights