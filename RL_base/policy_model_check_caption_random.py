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
from datetime import datetime
from open_flamingo import create_model_and_transforms
import argparse
import random
import numpy as np

logging.basicConfig(level=logging.INFO)


class COCOCaptionDataset(Dataset):
    def __init__(self, annotations_file, image_dir, split="train"):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            self.annotations = data['annotations']

        self.image_dir = image_dir
        self.split = split

        # 创建image_id到captions的映射
        self.image_to_captions = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_captions:
                self.image_to_captions[img_id] = []
            self.image_to_captions[img_id].append(ann['caption'])

        # 获取唯一的图像ID列表
        self.image_ids = list(self.image_to_captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 构建图像文件名
        if self.split == "train":
            image_filename = f'COCO_train2014_{str(image_id).zfill(12)}.jpg'
        else:
            image_filename = f'COCO_val2014_{str(image_id).zfill(12)}.jpg'

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        captions = self.image_to_captions[image_id]

        return {
            'image': image,
            'captions': captions,
            'image_id': image_id,
            'file_name': image_filename
        }


def custom_collate(batch):
    images = [item['image'] for item in batch]
    captions = [item['captions'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    return {
        'image': images,
        'captions': captions,
        'image_id': image_ids,
        'file_name': file_names
    }


def parse_args():
    parser = argparse.ArgumentParser()
    # ofv2相关参数
    parser.add_argument('--lm_path', type=str,
                        default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str,
                        default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str,
                        default="/data16tb/ljq/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='1')

    # 基础参数
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--BENCHMARK', default='coco_caption', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--shot_number',
                        type=int,
                        default=4,
                        help='Prompt num')

    # 数据集路径
    parser.add_argument('--train_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa/train2014')
    parser.add_argument('--val_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa/val2014')
    # parser.add_argument('--annotations_path', type=str,
    #                     default='/data16tb/ljq/datasets/coco_caption/annotations_captions_val2014.json')
    parser.add_argument('--val_json_file', type=str,
                        default='/data16tb/ljq/datasets/coco_caption/captions_val2014.json')
    parser.add_argument('--train_json_file', type=str,
                        default='/data16tb/ljq/datasets/coco_caption/captions_train2014.json')

    # 特征文件路径
    parser.add_argument('--features_file', type=str,
                        default='/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/coco_caption/features')

    # 输出路径
    parser.add_argument('--output_root', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear')

    args = parser.parse_args()

    # 设置输出路径
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'evaluation_results'), exist_ok=True)

    return args


def extract_and_save_features(clip_model, dataloader, output_file, device, preprocess):
    """提取并保存图像特征"""
    all_features = []
    all_image_ids = []
    all_file_names = []
    all_captions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            # 处理图像
            images = torch.stack([preprocess(img) for img in batch['image']]).to(device)
            features = clip_model.encode_image(images).float()

            all_features.append(features.cpu())
            all_image_ids.extend(batch['image_id'])
            all_file_names.extend(batch['file_name'])
            all_captions.extend(batch['captions'])

    all_features = torch.cat(all_features, dim=0)

    # 保存特征和元数据
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=all_features.numpy())
        f.create_dataset('image_ids', data=np.array(all_image_ids))
        f.create_dataset('file_names', data=np.array(all_file_names, dtype=h5py.special_dtype(vlen=str)))

        # 保存captions（可能长度不一）
        dt = h5py.special_dtype(vlen=str)
        captions_dataset = f.create_dataset('captions', shape=(len(all_captions),), dtype=dt)
        for i, caps in enumerate(all_captions):
            captions_dataset[i] = json.dumps(caps)  # 将caption列表转换为JSON字符串


def match_samples(args, device):
    """匹配查询样本和训练样本"""
    print("加载特征文件...")

    # 加载训练集和验证集特征
    with h5py.File(args.features_file + "_train.h5", 'r') as f:
        train_features = torch.from_numpy(f['features'][:]).to(device)
        train_image_ids = f['image_ids'][:]
        train_file_names = f['file_names'][:]
        train_captions = [json.loads(caps) for caps in f['captions'][:]]

    with h5py.File(args.features_file + "_val.h5", 'r') as f:
        val_features = torch.from_numpy(f['features'][:]).to(device)
        val_image_ids = f['image_ids'][:]
        val_file_names = f['file_names'][:]
        val_captions = [json.loads(caps) for caps in f['captions'][:]]

    results = []
    batch_size = 10
    k = args.shot_number  # 每个查询样本匹配的示例数量

    # 分批处理验证集样本
    for i in tqdm(range(0, len(val_features), batch_size), desc="匹配样本"):
        batch_end = min(i + batch_size, len(val_features))
        val_batch = val_features[i:batch_end]

        # 计算相似度
        sim = F.cosine_similarity(val_batch.unsqueeze(1), train_features.unsqueeze(0), dim=2)
        topk_scores, topk_indices = torch.topk(sim, k=k, dim=1)

        # 保存匹配结果
        for j, (scores, indices) in enumerate(zip(topk_scores, topk_indices)):
            val_idx = i + j

            # 修改为与eval格式一致的结构
            query_result = {
                'query_image_id': int(val_image_ids[val_idx]),
                'query_file_name': val_file_names[val_idx],
                'query_captions': val_captions[val_idx],
                'best_examples': []
            }

            # 添加最佳匹配的示例
            for score, train_idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
                example = {
                    'example_image_id': int(train_image_ids[train_idx]),
                    'example_file_name': train_file_names[train_idx],
                    'example_captions': train_captions[train_idx],
                    'similarity_score': float(score)  # 改用 similarity_score 作为键名
                }
                query_result['best_examples'].append(example)

            results.append(query_result)

    return results

def random_match_samples(args, device):
    """随机匹配查询样本和训练样本"""
    print("加载特征文件...")

    # 加载训练集和验证集特征
    with h5py.File(args.features_file + "_train.h5", 'r') as f:
        train_features = torch.from_numpy(f['features'][:]).to(device)
        train_image_ids = f['image_ids'][:]
        train_file_names = f['file_names'][:]
        train_captions = [json.loads(caps) for caps in f['captions'][:]]

    with h5py.File(args.features_file + "_val.h5", 'r') as f:
        val_features = torch.from_numpy(f['features'][:]).to(device)
        val_image_ids = f['image_ids'][:]
        val_file_names = f['file_names'][:]
        val_captions = [json.loads(caps) for caps in f['captions'][:]]

    results = []
    k = args.shot_number  # 每个查询样本匹配的示例数量
    train_indices = list(range(len(train_features)))  # 创建训练样本索引列表

    # 处理每个验证集样本
    for val_idx in tqdm(range(len(val_features)), desc="随机匹配样本"):
        # 随机选择k个训练样本
        selected_indices = random.sample(train_indices, k)

        query_result = {
            'query_image_id': int(val_image_ids[val_idx]),
            'query_file_name': val_file_names[val_idx],
            'query_captions': val_captions[val_idx],
            'best_examples': []
        }

        # 添加随机选择的示例
        for train_idx in selected_indices:
            example = {
                'example_image_id': int(train_image_ids[train_idx]),
                'example_file_name': train_file_names[train_idx],
                'example_captions': train_captions[train_idx],
                'similarity_score': 0.0  # 由于是随机匹配，将相似度设为0
            }
            query_result['best_examples'].append(example)

        results.append(query_result)

    return results

# 保存结果时添加类型检查
def ensure_serializable(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)

# 主函数中的修改
# 修改主函数中的文件保存逻辑
if __name__ == "__main__":
    args = parse_args()
    output_file = f"{args.BENCHMARK}_test_train_matching_random.json"
    print(f"输出文件: {output_file}")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # 创建数据加载器
    train_dataset = COCOCaptionDataset(args.train_json_file, args.train_image_dir, "train")
    val_dataset = COCOCaptionDataset(args.val_json_file, args.val_image_dir, "val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=custom_collate)

    # # 修改特征保存路径
    # os.makedirs(os.path.dirname(args.features_file), exist_ok=True)
    # extract_and_save_features(clip_model, train_loader,
    #                           args.features_file + "_train.h5", device, preprocess)
    # extract_and_save_features(clip_model, val_loader,
    #                           args.features_file + "_val.h5", device, preprocess)

    # 执行样本匹配
    matched_results = random_match_samples(args, device)
    # 确保所有数据都是可序列化的
    matched_results = ensure_serializable(matched_results)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_results, f, ensure_ascii=False, indent=4)

    print(f"匹配结果已保存到 {output_file}")