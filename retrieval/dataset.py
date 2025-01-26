import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        参数:
            root_dir (str): 数据集根目录路径
            split (str): 'train' 或 'val'
            transform: 图像预处理转换
        """
        self.root_dir = root_dir
        self.split = split
        # self.transform = transform

        # 读取类别ID
        with open(os.path.join(root_dir, 'wnids.txt')) as f:
            self.class_ids = [x.strip() for x in f.readlines()]

        # 读取类别描述文本
        self.class_descriptions = {}
        with open(os.path.join(root_dir, 'words.txt')) as f:
            for line in f:
                class_id, description = line.strip().split('\t')
                self.class_descriptions[class_id] = description

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_ids)}

        # 构建图像和标注路径
        self.images = []
        self.boxes = []
        self.labels = []
        self.descriptions = []  # 新增: 存储类别描述

        if split == 'train':
            # 处理训练集
            for class_id in self.class_ids:
                class_dir = os.path.join(root_dir, 'train', class_id)
                box_file = os.path.join(class_dir, f'{class_id}_boxes.txt')

                # 获取该类别的描述文本
                class_description = self.class_descriptions.get(class_id, '')

                # 读取边界框标注
                with open(box_file) as f:
                    for line in f:
                        if line.strip():
                            img_name, *box = line.strip().split()
                            img_path = os.path.join(class_dir, "images/" + img_name)
                            if os.path.exists(img_path):
                                self.images.append(img_path)
                                self.boxes.append([float(x) for x in box])
                                self.labels.append(self.class_to_idx[class_id])
                                self.descriptions.append(class_description)  # 添加类别描述

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # 获取边界框、标签和描述
        box = torch.tensor(self.boxes[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        description = self.descriptions[idx]

        # if self.transform:
        #     image = self.transform(image)

        return {
            'img_path':img_path,
            'image': image,
            'box': box,
            'label': label,
            'description': description  # 新增: 返回类别描述
        }