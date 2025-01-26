import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
from collections import defaultdict
import h5py
import logging
import json
from PIL import Image
import UTILS as utils
import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.dataset import TinyImageNetDataset
# from dataset import TinyImageNetDataset
from open_flamingo.inference import ofv2_classification
from open_flamingo import create_model_and_transforms
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # OpenFlamingo模型参数
    parser.add_argument('--lm_path', type=str, default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str, default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str, default="/data16tb/ljq/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='3')

    # 数据集参数
    parser.add_argument('--root_dir', type=str, default='/data16tb/ljq/datasets/tiny-imagenet-200')
    parser.add_argument('--features_file', type=str, default='tiny_imagenet_features.h5')
    parser.add_argument('--k_samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_root', type=str, default='../log/ofv2_base/END_OUTPUT_linear')
    parser.add_argument('--BENCHMARK', default='tiny_imagenet', type=str,
                        help='dataset type:"coco_caption vizwiz, okvqa, tiny_imagenet')
    parser.add_argument('--selected_samples', type=str, default='/data16tb/ljq/Code/AIApplyTech/selected_samples.json')

    parser.add_argument('--load_similarity', type=bool,
                        default=True,
                        help='load_similarity')

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_root, args.BENCHMARK)
    utils.create_dir(args.output_dir)

    return args


def custom_collate(batch):
    images = [item['image'] for item in batch]
    boxes = torch.stack([item['box'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    descriptions = [item['description'] for item in batch]

    return {
        'image': images,
        'box': boxes,
        'label': labels,
        'description': descriptions
    }


def extract_and_save_features(clip_model, dataloader, args, device):
    """提取并保存图像和文本特征"""
    all_features = []
    all_labels = []
    all_descriptions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            # 使用CLIP提取图像特征
            images = torch.stack([preprocess(img) for img in batch['image']]).to(device)
            features = clip_model.encode_image(images).float()

            all_features.append(features.cpu())
            all_labels.extend(batch['label'].tolist())
            all_descriptions.extend(batch['description'])

    all_features = torch.cat(all_features, dim=0)

    with h5py.File(args.features_file, 'w') as f:
        f.create_dataset('features', data=all_features.numpy())
        f.create_dataset('labels', data=np.array(all_labels))
        f.create_dataset('descriptions', data=np.array(all_descriptions, dtype=h5py.special_dtype(vlen=str)))


def compute_classification_confidence(ofv2_model, query_image, query_labels, example_images, example_labels,
                                      tokenizer, image_processor, device):
    """计算分类置信度"""
    results = ofv2_classification(
        ofv2_model,
        device,
        query_image,
        query_labels,
        example_images,
        example_labels,
        tokenizer,
        image_processor,
    )
    return results["predictions"][0]['confidence']


# def compute_similarity_batched(features, batch_size=10):
#     """分批计算相似度矩阵以节省显存"""
#     num_samples = features.shape[0]
#     sim_matrix = torch.zeros((num_samples, num_samples), device=features.device)
#
#     for i in tqdm(range(0, num_samples, batch_size), desc="计算相似度"):
#         batch_end = min(i + batch_size, num_samples)
#         batch = features[i:batch_end]
#
#         # 计算当前批次与所有样本的相似度
#         sim = torch.mm(batch, features.t())
#         sim_matrix[i:batch_end] = sim
#
#     return sim_matrix

def compute_and_save_similarity(features, args, batch_size=4):
    """
    计算相似度矩阵并保存到文件
    Args:
        features: 输入特征
        args: 参数对象
        batch_size: 每次计算的批次大小
        chunk_size: 结果矩阵的分块大小
    """
    similarity_file = os.path.join(args.output_dir, 'similarity_matrix.pt')

    # 如果相似度矩阵文件已存在，直接加载
    if os.path.exists(similarity_file) and args.load_similarity:
        print(f"加载已存在的相似度矩阵: {similarity_file}")
        return torch.load(similarity_file, map_location='cpu')
    num_samples = features.shape[0]
    sim_matrix = torch.zeros((num_samples, num_samples), device='cpu')

    for i in tqdm(range(0, num_samples, batch_size), desc="计算相似度"):
        batch_end = min(i + batch_size, num_samples)
        batch = features[i:batch_end]
        sim = F.cosine_similarity(batch.unsqueeze(1), features.unsqueeze(0), dim=2)
        sim_matrix[i:batch_end] = sim

    # 保存最终的相似度矩阵
    print(f"保存相似度矩阵到: {similarity_file}")
    torch.save(sim_matrix, similarity_file)

    return sim_matrix

def sample_and_label_pairs(ofv2_model, dataloader, args, device, tokenizer, image_processor):
    """为查询-示例样本对打分"""
    all_results = defaultdict(list)
    dataset = dataloader.dataset

    # 加载特征
    with h5py.File(args.features_file, 'r') as f:
        features = torch.from_numpy(f['features'][:]).to(device)
        labels = f['labels'][:]
        descriptions = f['descriptions'][:]

    # 计算相似度矩阵
    logging.info("计算样本间的相似度...")
    # similarity_matrix = compute_similarity_batched(features)
    similarity_matrix = compute_and_save_similarity(features, args)

    # # 获取所有类别名称
    # all_class_names = list(set(descriptions))

    if args.selected_samples is not None:
        # 加载选定的样本索引
        with open(args.selected_samples, 'r') as f:
            selected_samples_dict = json.load(f)
        selected_samples = []
        for key in selected_samples_dict:
            selected_samples.extend(selected_samples_dict[key])
        selected_samples = sorted(list(set(selected_samples)))  # 去重并排序
    else:
        selected_samples = range(len(dataset))

    with torch.no_grad():
        for query_idx in tqdm(selected_samples, desc="处理样本"):
            query_batch = dataset[query_idx]

            # 获取top-k相似样本
            sim_scores = similarity_matrix[query_idx]
            sim_scores[query_idx] = -1  # 排除自身
            top_k_values, top_k_indices = torch.topk(sim_scores, args.k_samples)

            # 处理每个相似样本
            for example_idx, similarity in zip(top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()):
                example_batch = dataset[example_idx]

                # 计算分类置信度
                confidence_score = compute_classification_confidence(
                    ofv2_model,
                    query_batch['image'],
                    query_batch['description'],
                    [example_batch['image']],
                    [example_batch['description']],
                    tokenizer,
                    image_processor,
                    device,
                )

                # 保存结果
                pair_info = {
                    'query_idx': int(query_idx),
                    'example_idx': int(example_idx),
                    'query_label': query_batch['description'],
                    'example_label': example_batch['description'],
                    'confidence_score': float(confidence_score),
                    'similarity_score': float(similarity),
                }
                all_results['samples'].append(pair_info)

            # 定期保存临时结果
            if query_idx % 100 == 0:
                save_temp_results(all_results, args)

    return all_results


def save_temp_results(results, args):
    """保存临时结果"""
    temp_file = os.path.join(args.output_dir, f"{args.BENCHMARK}_label_confidence_temp.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"临时结果已保存到 {temp_file}")


if __name__ == "__main__":
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载OpenFlamingo模型
    ofv2_model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path='ViT-L-14',
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.lm_tokenizer_path,
        cross_attn_every_n_layers=4,
        inference=True,
        precision='fp16',
        device=device,
        checkpoint_path=args.checkpoint_path,
    )

    # 加载数据集
    dataset = TinyImageNetDataset(args.root_dir, split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    # 加载CLIP模型用于特征提取
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 提取并保存特征
    if not os.path.exists(args.features_file):
        extract_and_save_features(clip_model, dataloader, args, device)

    # 运行样本对标注
    labeled_samples = sample_and_label_pairs(
        ofv2_model, dataloader, args, device, tokenizer, image_processor
    )

    # 保存最终结果
    output_file = os.path.join(args.output_dir, f"{args.BENCHMARK}_label_confidence_4800_{args.k_samples}.json")
    # output_file = os.path.join(args.output_dir, f"{args.BENCHMARK}_label_confidence_4800_{args.k_samples}_pro.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_samples, f, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file}")