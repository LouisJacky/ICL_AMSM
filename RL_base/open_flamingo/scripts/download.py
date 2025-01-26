from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch


# 首先下载模型权重
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")