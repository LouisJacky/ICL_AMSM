import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import clip
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_model import ResCaptionMatchingModel
from policy_model_check_caption import COCOCaptionDataset, custom_collate
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--shot_number', type=int, default=16)

    # 数据路径
    parser.add_argument('--train_image_dir', type=str,
                        default='/path/to/datasets/ok_vqa/train2014')
    parser.add_argument('--val_image_dir', type=str,
                        default='/path/to/datasets/ok_vqa/val2014')
    parser.add_argument('--train_json_file', type=str,
                        default='/path/to/datasets/coco_caption/captions_train2014.json')
    parser.add_argument('--val_json_file', type=str,
                        default='/path/to/datasets/coco_caption/captions_val2014.json')

    # 模型相关
    parser.add_argument('--policy_feature_dim', type=int, default=512)
    parser.add_argument('--policy_model_checkpoint', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear/coco_caption/policy_model_epoch_2.pth')

    args = parser.parse_args()
    return args


def load_trained_policy_model(checkpoint_path, device, args):
    """加载训练好的策略模型"""
    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 初始化策略模型
    policy_model = ResCaptionMatchingModel(clip_model, hidden_size=args.policy_feature_dim).to(device)

    # 加载保存的模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    policy_model.eval()

    return policy_model, preprocess


def save_method_results(results, method_name, args):
    """保存单个方法的结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "caption_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'metadata': {
            'method': method_name,
            'sample_size': args.sample_size,
            'shot_number': args.shot_number,
            'timestamp': timestamp
        },
        'results': results
    }

    output_file = os.path.join(output_dir, f'{method_name}_matching_{timestamp}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"{method_name}方法的结果已保存到: {output_file}")


def get_multimodal_matches(policy_model, test_samples, train_dataloader, args, device, preprocess):
    """使用多模态策略模型进行匹配"""
    results = {}
    train_samples = {}

    # 预处理训练数据
    for batch in tqdm(train_dataloader, desc="预处理训练数据"):
        for i in range(len(batch['image_id'])):
            train_samples[batch['image_id'][i]] = {
                'image': batch['image'][i],
                'captions': batch['captions'][i],
                'file_name': batch['file_name'][i]
            }

    # 计算所有训练样本的特征
    train_features = []
    train_ids = []
    with torch.no_grad():
        for train_batch in tqdm(train_dataloader, desc="计算训练样本特征"):
            train_images = torch.stack([preprocess(img) for img in train_batch['image']]).to(device)
            batch_features = policy_model.encode_image(train_images)
            train_features.append(batch_features)
            train_ids.extend(train_batch['image_id'])
    train_features = torch.cat(train_features, dim=0)

    # 对测试样本进行匹配
    with torch.no_grad():
        for test_sample in tqdm(test_samples, desc="多模态匹配"):
            test_image = preprocess(test_sample['image']).unsqueeze(0).to(device)
            test_features = policy_model.encode_image(test_image)

            # 计算相似度并获取top-k
            scores = policy_model.batch_compute_similarity(test_features, train_features)
            topk_scores, topk_indices = torch.topk(scores, k=args.shot_number, dim=1)

            # 保存结果
            top_examples = []
            for j, train_idx in enumerate(topk_indices[0]):
                train_id = train_ids[train_idx]
                train_sample = train_samples[train_id]
                example_info = {
                    'score': float(topk_scores[0][j]),
                    'example_captions': train_sample['captions'],
                    'example_file_name': train_sample['file_name'],
                    'example_image_id': train_id
                }
                top_examples.append(example_info)

            results[test_sample['image_id']] = {
                'query_image_id': test_sample['image_id'],
                'query_file_name': test_sample['file_name'],
                'query_captions': test_sample['captions'],
                'top_examples': top_examples
            }

    save_method_results(results, 'multimodal', args)
    return results


def get_image_matches(clip_model, test_samples, train_dataloader, args, device, preprocess):
    """使用CLIP图像特征进行匹配"""
    results = {}
    train_samples = {}

    # 计算所有训练样本的图像特征
    train_features = []
    train_info = []

    print("计算训练样本的图像特征...")
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="处理训练图像"):
            train_images = torch.stack([preprocess(img) for img in batch['image']]).to(device)
            image_features = clip_model.encode_image(train_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            train_features.append(image_features)

            for i in range(len(batch['image_id'])):
                train_info.append({
                    'captions': batch['captions'][i],
                    'file_name': batch['file_name'][i],
                    'image_id': batch['image_id'][i]
                })

    train_features = torch.cat(train_features, dim=0)

    # 对每个测试样本进行匹配
    print("开始图像特征匹配...")
    with torch.no_grad():
        for test_sample in tqdm(test_samples, desc="图像匹配"):
            test_image = preprocess(test_sample['image']).unsqueeze(0).to(device)
            test_features = clip_model.encode_image(test_image)
            test_features = test_features / test_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarity = torch.mm(test_features, train_features.t())
            topk_scores, topk_indices = torch.topk(similarity[0], k=args.shot_number)

            # 保存结果
            top_examples = []
            for score, idx in zip(topk_scores, topk_indices):
                train_sample = train_info[idx]
                example_info = {
                    'score': float(score),
                    'example_captions': train_sample['captions'],
                    'example_file_name': train_sample['file_name'],
                    'example_image_id': train_sample['image_id']
                }
                top_examples.append(example_info)

            results[test_sample['image_id']] = {
                'query_image_id': test_sample['image_id'],
                'query_file_name': test_sample['file_name'],
                'query_captions': test_sample['captions'],
                'top_examples': top_examples
            }

    save_method_results(results, 'image', args)
    return results


def get_random_matches(test_samples, train_dataloader, args):
    """随机选择训练样本进行匹配"""
    results = {}
    train_samples = []

    # 收集所有训练样本
    print("收集训练样本...")
    for batch in tqdm(train_dataloader, desc="收集训练样本"):
        for i in range(len(batch['image_id'])):
            train_samples.append({
                'captions': batch['captions'][i],
                'file_name': batch['file_name'][i],
                'image_id': batch['image_id'][i]
            })

    total_train_samples = len(train_samples)

    # 对每个测试样本随机选择训练样本
    print("开始随机匹配...")
    for test_sample in tqdm(test_samples, desc="随机匹配"):
        # 随机选择示例
        random_indices = torch.randperm(total_train_samples)[:args.shot_number]

        top_examples = []
        for idx in random_indices:
            train_sample = train_samples[idx]
            example_info = {
                'score': 0.0,  # 随机方法不计算相似度分数
                'example_captions': train_sample['captions'],
                'example_file_name': train_sample['file_name'],
                'example_image_id': train_sample['image_id']
            }
            top_examples.append(example_info)

        results[test_sample['image_id']] = {
            'query_image_id': test_sample['image_id'],
            'query_file_name': test_sample['file_name'],
            'query_captions': test_sample['captions'],
            'top_examples': top_examples
        }

    save_method_results(results, 'random', args)
    return results

def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载CLIP模型和策略模型
    policy_model, preprocess = load_trained_policy_model(args.policy_model_checkpoint, device, args)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 创建数据加载器
    val_dataset = COCOCaptionDataset(args.val_json_file, args.val_image_dir, split="val")
    train_dataset = COCOCaptionDataset(args.train_json_file, args.train_image_dir, split="train")

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                num_workers=args.num_workers, collate_fn=custom_collate)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, collate_fn=custom_collate)

    # 随机采样测试样本
    total_samples = len(val_dataloader.dataset)
    # 生成随机索引
    if args.sample_size is not None:
        random_indices = torch.randperm(total_samples)[:min(args.sample_size, total_samples)]
    else:
        random_indices = torch.arange(total_samples)

    test_samples = []
    for idx in random_indices:
        batch = val_dataloader.dataset[idx.item()]
        sample = {k: v if isinstance(v, (list, torch.Tensor)) else v
                  for k, v in batch.items()}
        test_samples.append(sample)

    try:
        # 获取并保存多模态匹配结果
        print("开始多模态匹配...")
        multimodal_results = get_multimodal_matches(policy_model, test_samples, train_dataloader,
                                                    args, device, preprocess)

        # 获取并保存图像匹配结果
        print("开始图像匹配...")
        image_results = get_image_matches(clip_model, test_samples, train_dataloader,
                                          args, device, preprocess)

        # 获取并保存随机匹配结果
        print("开始随机匹配...")
        random_results = get_random_matches(test_samples, train_dataloader, args)

        print("所有分析完成！")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()