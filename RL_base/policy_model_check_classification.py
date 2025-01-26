import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import clip
import torch.nn.functional as F
from tqdm import tqdm
import h5py
import logging
import argparse
import random
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split)

        self.images = []
        self.labels = []
        self.file_names = []

        if split == 'train':
            self.classes = sorted(os.listdir(self.image_dir))
            for class_idx, class_name in enumerate(self.classes):
                class_dir = os.path.join(self.image_dir, class_name, "images")
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            self.images.append(os.path.join(class_name, "images", img_name))
                            self.labels.append(class_name)
                            self.file_names.append(img_name)
        else:
            val_annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    img_name, class_name, *_ = line.strip().split('\t')
                    self.images.append(os.path.join("images", img_name))
                    self.labels.append(class_name)
                    self.file_names.append(img_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        return {
            'image': image,
            'label': self.labels[idx],
            'file_name': self.images[idx]
        }


def custom_collate(batch):
    if not batch:
        return None

    return {
        'image': [item['image'] for item in batch],
        'label': [item['label'] for item in batch],
        'file_name': [item['file_name'] for item in batch]
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/path/to/datasets/tiny-imagenet-200')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shot_number', type=int, default=4)
    parser.add_argument('--shot_per_class', type=int, default=2)
    parser.add_argument('--output_root', type=str, default='../log/ofv2_base/END_OUTPUT_linear')
    parser.add_argument('--BENCHMARK', default='tiny_imagenet')
    parser.add_argument('--features_dir', type=str, default='features')

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, args.features_dir), exist_ok=True)

    return args


def extract_and_save_features(clip_model, dataloader, output_file, device, preprocess):
    """提取并保存图像特征"""
    all_features = []
    all_labels = []
    all_file_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            if batch is None:
                continue

            images = torch.stack([preprocess(img) for img in batch['image']]).to(device)
            features = clip_model.encode_image(images).float()

            all_features.append(features.cpu())
            all_labels.extend(batch['label'])
            all_file_names.extend(batch['file_name'])

    all_features = torch.cat(all_features, dim=0)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=all_features.numpy())
        f.create_dataset('labels', data=np.array(all_labels, dtype=h5py.special_dtype(vlen=str)))
        f.create_dataset('file_names', data=np.array(all_file_names, dtype=h5py.special_dtype(vlen=str)))

def random_match_samples(args, device):
    """随机匹配查询样本和训练样本"""
    print("加载特征文件...")

    features_path = os.path.join(args.output_path, args.features_dir)

    with h5py.File(os.path.join(features_path, "train_features.h5"), 'r') as f:
        train_features = torch.from_numpy(f['features'][:]).to(device)
        train_labels = [label.decode('utf-8') for label in f['labels'][:]]
        train_file_names = [name.decode('utf-8') for name in f['file_names'][:]]

    with h5py.File(os.path.join(features_path, "val_features.h5"), 'r') as f:
        val_features = torch.from_numpy(f['features'][:]).to(device)
        val_labels = [label.decode('utf-8') for label in f['labels'][:]]
        val_file_names = [name.decode('utf-8') for name in f['file_names'][:]]

    results = []
    k = args.shot_number
    train_indices = list(range(len(train_features)))

    for val_idx in tqdm(range(len(val_features)), desc="随机匹配样本"):
        # 随机选择k个训练样本
        selected_indices = random.sample(train_indices, k)

        query_result = {
            'query_file_name': val_file_names[val_idx],
            'query_label': val_labels[val_idx],
            'best_examples': []
        }

        # 添加随机选择的示例
        for train_idx in selected_indices:
            example = {
                'example_file_name': train_file_names[train_idx],
                'example_label': train_labels[train_idx],
                'similarity_score': 0.0  # 随机匹配将相似度设为0
            }
            query_result['best_examples'].append(example)

        results.append(query_result)

    return results

def match_samples(args, device):
    """匹配查询样本和训练样本"""
    print("加载特征文件...")

    features_path = os.path.join(args.output_path, args.features_dir)

    with h5py.File(os.path.join(features_path, "train_features.h5"), 'r') as f:
        train_features = torch.from_numpy(f['features'][:]).to(device)
        # 将bytes转换为字符串
        train_labels = [label.decode('utf-8') for label in f['labels'][:]]
        train_file_names = [name.decode('utf-8') for name in f['file_names'][:]]

    with h5py.File(os.path.join(features_path, "val_features.h5"), 'r') as f:
        val_features = torch.from_numpy(f['features'][:]).to(device)
        # 将bytes转换为字符串
        val_labels = [label.decode('utf-8') for label in f['labels'][:]]
        val_file_names = [name.decode('utf-8') for name in f['file_names'][:]]

    results = []
    batch_size = 10
    k = args.shot_number

    for i in tqdm(range(0, len(val_features), batch_size), desc="匹配样本"):
        batch_end = min(i + batch_size, len(val_features))
        val_batch = val_features[i:batch_end]

        # 计算余弦相似度
        sim = F.cosine_similarity(val_batch.unsqueeze(1), train_features.unsqueeze(0), dim=2)
        #
        # # 获取每个查询样本的top-k个最相似训练样本
        # topk_scores, topk_indices = torch.topk(sim, k=k, dim=1)
        #
        # for j, (scores, indices) in enumerate(zip(topk_scores, topk_indices)):
        #     val_idx = i + j
        #
        #     query_result = {
        #         'query_file_name': val_file_names[val_idx],
        #         'query_label': val_labels[val_idx],
        #         'best_examples': []
        #     }
        #
        #     # 添加最佳匹配的示例
        #     for score, train_idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
        #         example = {
        #             'example_file_name': train_file_names[train_idx],
        #             'example_label': train_labels[train_idx],
        #             'similarity_score': float(score)
        #         }
        #         query_result['best_examples'].append(example)
        #
        #     results.append(query_result)

        for j in range(val_batch.size(0)):
            val_idx = i + j
            query_label = val_labels[val_idx]

            # same_idx_map = []
            # diff_idx_map = []
            #
            # # 1. 获取相同类别的最相似样本
            # same_class_mask = torch.zeros_like(sim[j], dtype=torch.bool)
            # diff_class_mask = torch.zeros_like(sim[j], dtype=torch.bool)
            # for idx, train_label in enumerate(train_labels):
            #     if train_label == query_label:
            #         same_class_mask[idx] = True
            #         same_idx_map.append(idx)
            #     else:
            #         diff_class_mask[idx] = True
            #         diff_idx_map.append(idx)

            # 1. 获取相同类别和不同类别的掩码
            train_labels_array = np.array(train_labels)
            same_class_mask = (train_labels_array == query_label)
            diff_class_mask = ~same_class_mask

            # 获取相同类别和不同类别的索引
            same_idx_map = np.where(same_class_mask)[0]
            diff_idx_map = np.where(diff_class_mask)[0]

            # 转换为torch的布尔掩码
            same_class_mask = torch.from_numpy(same_class_mask).to(device)
            diff_class_mask = torch.from_numpy(diff_class_mask).to(device)

            same_class_sim = sim[j][same_class_mask]
            top1_same_score, top1_same_idx = torch.topk(same_class_sim, k=args.shot_per_class)
            for k in range(len(top1_same_idx)):
                top1_same_idx[k] = same_idx_map[top1_same_idx[k]]

            # 2. 获取不同类别的最相似样本
            diff_class_sim = sim[j][diff_class_mask]
            topk_diff_scores, topk_diff_indices = torch.topk(diff_class_sim, k=args.shot_number)
            for k in range(len(topk_diff_indices)):
                topk_diff_indices[k] = diff_idx_map[topk_diff_indices[k]]

            # 3. 合并所有候选样本并重新排序
            all_scores = torch.cat([top1_same_score, topk_diff_scores])
            all_indices = torch.cat([top1_same_idx, topk_diff_indices])

            # 从k+1个样本中选出top-k
            final_topk_scores, final_topk_idx = torch.topk(all_scores, k=args.shot_number)
            final_indices = all_indices[final_topk_idx]

            query_result = {
                'query_file_name': val_file_names[val_idx],
                'query_label': query_label,
                'best_examples': []
            }

            # 添加最佳匹配的示例
            for score, train_idx in zip(final_topk_scores.cpu().numpy(), final_indices.cpu().numpy()):
                example = {
                    'example_file_name': train_file_names[train_idx],
                    'example_label': train_labels[train_idx],
                    'similarity_score': float(score)
                }
                query_result['best_examples'].append(example)

            results.append(query_result)

    return results


if __name__ == "__main__":
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # 创建数据加载器
    train_dataset = TinyImageNetDataset(args.root_dir, split="train")
    val_dataset = TinyImageNetDataset(args.root_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    features_path = os.path.join(args.output_path, args.features_dir)

    # # 提取并保存特征
    # print("提取训练集特征...")
    # extract_and_save_features(
    #     clip_model,
    #     train_loader,
    #     os.path.join(features_path, "train_features.h5"),
    #     device,
    #     preprocess
    # )

    # print("提取验证集特征...")
    # extract_and_save_features(
    #     clip_model,
    #     val_loader,
    #     os.path.join(features_path, "val_features.h5"),
    #     device,
    #     preprocess
    # )

    # 执行样本匹配
    matched_results = match_samples(args, device)

    # 保存结果
    output_file = os.path.join(args.output_path, f"{args.BENCHMARK}_classification_matches_si_shot_per_class.json")

    # # 执行随机样本匹配
    # matched_results = random_match_samples(args, device)
    #
    # # 保存结果
    # output_file = os.path.join(args.output_path, f"{args.BENCHMARK}_classification_matches_random.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_results, f, ensure_ascii=False, indent=4)

    print(f"匹配结果已保存到 {output_file}")