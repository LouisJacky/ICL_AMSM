import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import clip

import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy_model import ResCaptionMatchingModel
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
from policy_model_label_caption_cider import COCOCaptionDataset, custom_collate


def parse_args():
    parser = argparse.ArgumentParser(description='Caption Policy Model Evaluation')

    # 模型相关参数
    parser.add_argument('--policy_feature_dim', type=int, default=512)
    parser.add_argument('--policy_model_checkpoint', type=str,
                        # default='../log/ofv2_base/END_OUTPUT_linear/coco_caption/policy_model_epoch_4.pth')
                        default='../log/ofv2_base/END_OUTPUT_linear/coco_caption/policy_model_epoch_2.pth',)

    # 数据集相关参数
    parser.add_argument('--val_json_file', type=str,
                        default='/data16tb/ljq/datasets/coco_caption/captions_val2014.json')
    parser.add_argument('--train_json_file', type=str,
                        default='/data16tb/ljq/datasets/coco_caption/captions_train2014.json')
    parser.add_argument('--val_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa')
    parser.add_argument('--train_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--shot_number', type=int, default=8)
    parser.add_argument('--output_root', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear')
    parser.add_argument('--BENCHMARK', default='coco_caption')

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    os.makedirs(args.output_path, exist_ok=True)

    return args


def load_trained_policy_model(checkpoint_path, device, args):
    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 初始化策略模型
    policy_model = ResCaptionMatchingModel(clip_model, hidden_size=args.policy_feature_dim).to(device)

    # 加载保存的模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])

    # 将模型设置为评估模式
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

            # 获取top-k个最相似的样本
            top_k_scores, top_k_indices = scores.topk(args.shot_number, dim=1)

            # 收集结果
            for i in range(len(val_batch['image'])):
                query_result = {
                    'query_image_id': val_batch['image_id'][i],
                    'query_file_name': val_batch['file_name'][i],
                    'query_captions': val_batch['captions'][i],
                    'best_examples': []
                }

                for j, (score, idx) in enumerate(zip(top_k_scores[i], top_k_indices[i])):
                    train_batch_idx = idx // args.batch_size
                    train_sample_idx = idx % args.batch_size
                    train_batch = train_data[train_batch_idx]

                    example = {
                        'example_image_id': train_batch['image_id'][train_sample_idx],
                        'example_file_name': train_batch['file_name'][train_sample_idx],
                        'example_captions': train_batch['captions'][train_sample_idx],
                        'similarity_score': score.item()
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
    val_dataset = COCOCaptionDataset(args.val_json_file, args.val_image_dir, split="val")
    train_dataset = COCOCaptionDataset(args.train_json_file, args.train_image_dir, split="train")

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