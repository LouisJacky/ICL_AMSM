import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import clip
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_model import ResCaptionMatchingModel
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split)

        # 收集所有图像和标签
        self.images = []
        self.labels = []
        self.file_names = []

        if split=='train':
            # 获取所有类别
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
            # 读取验证集标注文件
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
    parser = argparse.ArgumentParser(description='Classification Policy Model Evaluation')

    # 模型相关参数
    parser.add_argument('--policy_feature_dim', type=int, default=512)
    parser.add_argument('--policy_model_checkpoint', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear/tiny_imagenet/policy_model_epoch_2.pth')

    # 数据集相关参数
    parser.add_argument('--root_dir', type=str,
                        default='/data16tb/ljq/datasets/tiny-imagenet-200')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--shot_number', type=int, default=4)
    parser.add_argument('--shot_per_class', type=int, default=2)
    parser.add_argument('--output_root', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear')
    parser.add_argument('--BENCHMARK', default='tiny_imagenet')

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    os.makedirs(args.output_path, exist_ok=True)

    return args


def load_trained_policy_model(checkpoint_path, device, args):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    policy_model = ResCaptionMatchingModel(clip_model, hidden_size=args.policy_feature_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    policy_model.eval()
    return policy_model, preprocess


def evaluate_policy_model(policy_model, val_dataloader, train_dataloader, args, device, preprocess):
    policy_model.eval()
    all_results = []

    # 预计算所有训练样本的特征
    print("计算训练集特征...")
    train_features = []
    train_data = []

    with torch.no_grad():
        for train_batch in tqdm(train_dataloader, desc="处理训练样本"):
            if train_batch is None:
                continue
            train_images = torch.stack([preprocess(img) for img in train_batch['image']]).to(device)
            batch_features = policy_model.encode_image(train_images)
            train_features.append(batch_features)
            train_data.append(train_batch)

    train_features = torch.cat(train_features, dim=0)

    print("评估验证集...")
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc="处理验证样本"):
            if val_batch is None:
                continue

            val_images = torch.stack([preprocess(img) for img in val_batch['image']]).to(device)
            val_features = policy_model.encode_image(val_images)

            # 计算与所有训练样本的相似度
            scores = policy_model.batch_compute_similarity(val_features, train_features)

            # # 获取top-k个最相似的样本
            # top_k_scores, top_k_indices = scores.topk(args.shot_number, dim=1)
            #
            # # 收集结果
            # for i in range(len(val_batch['image'])):
            #     query_result = {
            #         'query_file_name': val_batch['file_name'][i],
            #         'query_label': val_batch['label'][i],
            #         'best_examples': []
            #     }
            #
            #     for j, (score, idx) in enumerate(zip(top_k_scores[i], top_k_indices[i])):
            #         train_batch_idx = idx // args.batch_size
            #         train_sample_idx = idx % args.batch_size
            #         train_batch = train_data[train_batch_idx]
            #
            #         example = {
            #             'example_file_name': train_batch['file_name'][train_sample_idx],
            #             'example_label': train_batch['label'][train_sample_idx],
            #             'match_score': score.item()
            #         }
            #         query_result['best_examples'].append(example)
            #
            #     all_results.append(query_result)

            for i in range(len(val_batch['image'])):
                query_label = val_batch['label'][i]

                # 1. 获取相同类别和不同类别的掩码
                train_labels_array = np.array([item for batch in train_data for item in batch['label']])
                same_class_mask = (train_labels_array == query_label)
                diff_class_mask = ~same_class_mask

                # 获取相同类别和不同类别的索引
                same_idx_map = np.where(same_class_mask)[0]
                diff_idx_map = np.where(diff_class_mask)[0]

                # 转换为torch的布尔掩码
                same_class_mask = torch.from_numpy(same_class_mask).to(device)
                diff_class_mask = torch.from_numpy(diff_class_mask).to(device)

                # 获取相同类别的最相似样本
                same_class_sim = scores[i][same_class_mask]
                top1_same_score, top1_same_idx = torch.topk(same_class_sim, k=args.shot_per_class)
                top1_same_idx = torch.tensor([same_idx_map[idx] for idx in top1_same_idx.cpu().numpy()]).to(device)

                # 获取不同类别的最相似样本
                diff_class_sim = scores[i][diff_class_mask]
                topk_diff_scores, topk_diff_indices = torch.topk(diff_class_sim, k=args.shot_number)
                topk_diff_indices = torch.tensor([diff_idx_map[idx] for idx in topk_diff_indices.cpu().numpy()]).to(
                    device)

                # 合并所有候选样本并重新排序
                all_scores = torch.cat([top1_same_score, topk_diff_scores])
                all_indices = torch.cat([top1_same_idx, topk_diff_indices])

                # 从k+1个样本中选出top-k
                final_topk_scores, final_topk_idx = torch.topk(all_scores, k=args.shot_number)
                final_indices = all_indices[final_topk_idx]

                query_result = {
                    'query_file_name': val_batch['file_name'][i],
                    'query_label': query_label,
                    'best_examples': []
                }

                # 添加最佳匹配的示例
                for score, idx in zip(final_topk_scores.cpu().numpy(), final_indices.cpu().numpy()):
                    train_batch_idx = idx // args.batch_size
                    train_sample_idx = idx % args.batch_size
                    train_batch = train_data[train_batch_idx]

                    example = {
                        'example_file_name': train_batch['file_name'][train_sample_idx],
                        'example_label': train_batch['label'][train_sample_idx],
                        'match_score': float(score)
                    }
                    query_result['best_examples'].append(example)

                all_results.append(query_result)

    return all_results


if __name__ == "__main__":
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    policy_model, preprocess = load_trained_policy_model(args.policy_model_checkpoint, device, args)

    # 创建数据加载器
    val_dataset = TinyImageNetDataset(args.root_dir, split="val")
    train_dataset = TinyImageNetDataset(args.root_dir, split="train")

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=custom_collate)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  collate_fn=custom_collate)

    # 进行评估
    results = evaluate_policy_model(policy_model, val_dataloader, train_dataloader,
                                    args, device, preprocess)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_path, f'eval_results_{timestamp}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"评估完成，结果已保存至: {output_file}")
    print(f"共处理了 {len(results)} 个验证集样本")