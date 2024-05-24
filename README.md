# OpenView: Cross-Modal 3D Representation with Multi-View Images and Point Clouds
##### State: The paper is under review.

##### Code of " OpenView: Cross-Modal 3D Representation with Multi-View Images and Point Clouds "

## Checkpoints

Due to the difficulty of sharing large files anonymously, the model checkpoints and test data will be released when the paper is published.

## Installation

If you would to run the testing or (and) training locally, you may need to install the dependendices.

1. Create a conda environment and install [pytorch](https://pytorch.org/get-started/previous-versions/), [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/quick_start.html), and [DGL](https://www.dgl.ai/pages/start.html) by the following commands or their official guides:
```
conda create -n OpenView python=3.9
conda activate OpenView
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
conda install -c dglteam/label/cu113 dgl
```
2. Install the following packages:
```
pip install huggingface_hub wandb omegaconf torch_redstone einops tqdm open3d open_clip_torch
```

## Testing

We provide test code on three datasets Objaverse, Omniobject3d, ModelNet40 for retrieval task and classification task.

```
python test_retrieval_objaverse.py
python test_retrieval_omniobject3d.py
python test_classification_modelnet.py
python test_classification_objaverse.py
python test_classification_omniobject3d.py
```
## Training

1. The processed training data can be found [here](https://huggingface.co/datasets/OpenShape/openshape-training-data), which are collected and processed by  [OpenShape](https://github.com/Colin97/OpenShape_code/).
2. Run the training by the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py
```


