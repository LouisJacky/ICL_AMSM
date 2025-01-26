from open_flamingo import create_model_and_transforms
import torch
from PIL import Image
import requests
import json
import random
import os
from collections import defaultdict
import re
import argparse
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from cococaption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cococaption.pycocoevalcap.bleu.bleu import Bleu
from cococaption.pycocoevalcap.meteor.meteor import Meteor
from cococaption.pycocoevalcap.rouge.rouge import Rouge
from cococaption.pycocoevalcap.cider.cider import Cider
import UTILS as utils
import h5py
import torch.nn.functional as F
import clip
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from retrieval.process_label_caption_cider import extract_short_answer

def parse_args():
    parser = argparse.ArgumentParser(description='COCO Caption Generation and Evaluation')

    # 路径相关参数
    parser.add_argument('--lm_path', default='/data16tb/ljq/checkpoints/mpt-7b',
                        help='语言模型路径')
    parser.add_argument('--lm_tokenizer_path', default='/data16tb/ljq/checkpoints/mpt-7b',
                        help='分词器路径')
    parser.add_argument('--checkpoint_path', default='/data16tb/ljq/checkpoints/ofv2/checkpoint.pt',
                        help='检查点路径')
    parser.add_argument('--json_path', default='/data16tb/ljq/datasets/coco_caption/captions_train2014.json',
                        help='COCO标注JSON文件路径')
    parser.add_argument('--annotations_path',
                        default='/data16tb/ljq/datasets/coco_caption/annotations_captions_train2014.json',
                        help='COCO评估标注文件路径')
    parser.add_argument('--image_dir', default='/data16tb/ljq/datasets/ok_vqa',
                        help='图像根目录')
    parser.add_argument('--load_similarity', type=bool,
                        default=True,
                        help='load_similarity')

    # 其他参数
    parser.add_argument('--device', default='cuda:6', help='使用的设备')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--output_root', type=str, default='../log/ofv2_base/END_OUTPUT_linear')
    parser.add_argument('--BENCHMARK', default='coco_caption', type=str,
                        help='dataset type:"coco_caption vizwiz, okvqa')

    # 添加新参数
    parser.add_argument('--features_file', type=str,
                        default='coco_features.h5',
                        help='Path to save/load image features')
    parser.add_argument('--k_samples', type=int,
                        default=32,
                        help='Number of top similar samples to select')
    # 添加选定样本文件路径参数
    parser.add_argument('--selected_samples', type=str,
                        default='/data16tb/ljq/Code/ICL_diversity_ofv3/retrieval/selected_coco_samples.json',
                        help='选定样本的JSON文件路径')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='要处理的样本数量,在selected_samples!=None时无效')


    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_root, args.BENCHMARK)
    utils.create_dir(args.output_dir)

    return args


class COCOCaptionDataset(Dataset):
    def __init__(self, json_file, image_dir, split="train"):
        """
        初始化COCO Caption数据集
        Args:
            json_file: 包含标注的JSON文件路径
            image_dir: 图像目录路径
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.images = data['images']
        self.annotations = data['annotations']
        self.image_dir = image_dir
        self.split = split

        # 创建image_id到annotations的映射
        self.image_to_captions = defaultdict(list)
        for ann in self.annotations:
            self.image_to_captions[ann['image_id']].append(ann['caption'])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']

        # 构建图像路径
        image_path = os.path.join(self.image_dir, f'{self.split}2014', image_info['file_name'])

        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {str(e)}")
            return None

        # 获取该图像的所有描述
        captions = self.image_to_captions[image_id]

        return {
            'image': image,
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'captions': captions,
            'image_path': image_path
        }


def custom_collate(batch):
    """
    自定义的数据批处理函数
    Args:
        batch: 数据批次
    Returns:
        处理后的批次数据
    """
    # 过滤掉None值
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return {
        'image': [item['image'] for item in batch],
        'image_id': [item['image_id'] for item in batch],
        'file_name': [item['file_name'] for item in batch],
        'captions': [item['captions'] for item in batch],
        'image_path': [item['image_path'] for item in batch]
    }


def get_coco_dataloader(json_file, image_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    创建COCO Caption数据加载器
    Args:
        json_file: JSON文件路径
        image_dir: 图像目录路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
    Returns:
        DataLoader实例
    """
    dataset = COCOCaptionDataset(json_file, image_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate
    )


def extract_and_save_features(dataloader, args, clip_model, device, preprocess):
    """批量提取并保存图像特征"""
    all_image_features = []
    all_image_ids = []

    for batch in tqdm(dataloader, desc="批量提取特征"):
        if batch is None:
            continue

        try:
            # 批量预处理图像
            batch_images = torch.stack([preprocess(img) for img in batch['image']])
            batch_images = batch_images.to(device)

            # 批量提取特征
            with torch.no_grad():
                batch_features = clip_model.encode_image(batch_images).float().detach()
                all_image_features.append(batch_features.cpu())
                all_image_ids.extend(batch['image_id'])

        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            # 如果批处理失败，尝试逐个处理该批次的图像
            for image, image_id in zip(batch['image'], batch['image_id']):
                try:
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input).float().detach()
                        all_image_features.append(image_features.cpu())
                        all_image_ids.append(image_id)
                except Exception as e:
                    print(f"处理单个图像时出错 ID {image_id}: {str(e)}")
                    continue

    # 合并所有特征
    all_image_features = torch.cat(all_image_features, dim=0)

    # 保存特征
    with h5py.File(args.features_file, 'w') as f:
        f.create_dataset('image_features', data=all_image_features.numpy())
        f.create_dataset('image_ids', data=np.array(all_image_ids))

        # 创建id到索引的映射
        id_to_idx = {str(img_id): idx for idx, img_id in enumerate(all_image_ids)}
        id_to_idx_group = f.create_group('id_to_idx')
        for img_id, idx in id_to_idx.items():
            id_to_idx_group.attrs[img_id] = idx

    return all_image_features, all_image_ids


# def compute_similarity_batched(features, batch_size=100):
#     """批量计算特征相似度"""
#     num_samples = features.shape[0]
#     sim_matrix = torch.zeros((num_samples, num_samples), device=features.device)
#
#     for i in tqdm(range(0, num_samples, batch_size), desc="计算相似度"):
#         batch_end = min(i + batch_size, num_samples)
#         batch = features[i:batch_end]
#         sim = F.cosine_similarity(batch.unsqueeze(1), features.unsqueeze(0), dim=2)
#         sim_matrix[i:batch_end] = sim
#
#     return sim_matrix

def compute_and_save_similarity(features, args, batch_size=10, chunk_size=20):
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

def convert_to_coco_format(results):
    """
    将结果转换为COCO评估格式
    """
    predictions = []
    for image_name, gens in results.items():
        img_id = int(image_name.split("_")[-1].split(".")[0])
        for gen in gens:
            predictions.append({
                "image_id": img_id,
                "caption": extract_short_answer(gen['generated_text'])
            })
    return predictions


# def calculate_scores(results, ground_truths, args):
#     """
#     计算评估分数
#     """
#     os.makedirs(args.output_dir, exist_ok=True)
#     res_file = os.path.join(args.output_dir, "results.json")
#
#     predictions = convert_to_coco_format(results)
#
#     with open(res_file, 'w') as f:
#         json.dump(predictions, f)
#
#     coco = COCO(args.annotations_path)
#     cocoRes = coco.loadRes(res_file)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.params['image_id'] = cocoRes.getImgIds()
#     cocoEval.evaluate()
#
#     return cocoEval.eval


def calculate_cider_score(results, args):
    """
    计算每个样本对的CIDEr分数
    返回: dict - 图片ID到分数的映射
    """
    predictions = convert_to_coco_format(results)
    coco = COCO(args.annotations_path)

    gts = {}
    res = {}
    img_id_to_pred = {}  # 用于追踪每个预测对应的图片ID

    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in res:
            res[img_id] = []
        res[img_id].append(pred['caption'])
        # 保存预测与图片ID的对应关系
        img_id_to_pred[img_id] = pred['caption']

    for img_id in res.keys():
        gts[img_id] = []
        anns = coco.imgToAnns[img_id]
        for ann in anns:
            gts[img_id].append(ann['caption'])

    scorer = Cider()
    _, scores = scorer.compute_score(gts, res)

    # 将分数与图片ID对应
    id_to_score = {}
    for img_id, score in zip(res.keys(), scores):
        id_to_score[img_id] = float(score)  # 转换numpy类型为Python原生类型

    return id_to_score


def process_sample_pair(model, demo_sample, query_sample, tokenizer, image_processor, device):
    """处理样本对并生成描述"""
    try:
        demo_image = Image.open(demo_sample['image_path'])
        query_image = Image.open(query_sample['image_path'])

        demo_caption = random.choice(demo_sample['captions'])
        prompt = f"<image>Output:{demo_caption}<|endofchunk|><image>Output:"

        tokenizer.padding_side = "left"
        lang_x = tokenizer([prompt], return_tensors="pt")

        vision_x = [image_processor(demo_image).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        vision_x = vision_x.to(device).half()
        input_ids = lang_x["input_ids"].to(device)
        attention_mask = lang_x["attention_mask"].to(device)

        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            num_beams=3,
        )

        return {
            'demo_caption': demo_caption,
            'generated_text': tokenizer.decode(generated_text[0]),
        }

    except Exception as e:
        print(f"处理样本时出错: {str(e)}")
        return None


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("正在加载模型...")
    # 初始化Flamingo模型
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path='ViT-L-14',
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.lm_tokenizer_path,
        cross_attn_every_n_layers=4,
        inference=True,
        precision='fp16',
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )
    # 初始化CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 创建数据加载器
    print("正在加载数据...")
    dataloader = get_coco_dataloader(
        args.json_path,
        args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataset = dataloader.dataset

    # 提取或加载特征
    print("正在处理图像特征...")
    if not os.path.exists(args.features_file):
        image_features, image_ids = extract_and_save_features(dataloader, args, clip_model, device, preprocess)
    else:
        with h5py.File(args.features_file, 'r') as f:
            image_features = torch.from_numpy(f['image_features'][:]).to(device)
            image_ids = f['image_ids'][:]

    print("正在计算相似度矩阵...")
    # 计算或加载相似度矩阵
    similarity_matrix = compute_and_save_similarity(
        image_features,
        args,
    )

    # 选择要处理的样本
    if args.selected_samples is not None:
        # 加载选定的样本信息
        with open(args.selected_samples, 'r') as f:
            selected_data = json.load(f)

        # 提取所有选定的image_ids
        selected_image_ids = [
            sample_info['image_id']
            for sample_info in selected_data['samples'].values()
        ]

        # 找到dataset中对应的索引
        query_indices = []
        image_id_to_idx = {item['image_id']: idx for idx, item in enumerate(dataset)}

        for image_id in selected_image_ids:
            if image_id in image_id_to_idx:
                query_indices.append(image_id_to_idx[image_id])

        print(f"找到 {len(query_indices)} 个匹配的查询样本")

    elif args.num_samples is not None:
        query_indices = random.sample(range(len(dataset)), args.num_samples)
    else:
        query_indices = range(len(dataset))

    # 初始化结果存储
    generation_results = []

    # 添加总体进度条
    total_iterations = len(query_indices) * args.k_samples
    progress_bar = tqdm(total=total_iterations, desc="总体处理进度")

    count_num_samples = 0

    # 处理每个查询样本
    for query_idx in query_indices:
        query_sample = dataset[query_idx]
        if query_sample is None:
            continue

        # 获取相似度分数并找到top-k个最相似样本
        sim_scores = similarity_matrix[query_idx]
        sim_scores[query_idx] = -1  # 排除自身
        top_k_values, top_k_indices = torch.topk(sim_scores, args.k_samples)

        # 处理每个相似样本
        for example_idx, similarity in zip(top_k_indices.numpy(), top_k_values.numpy()):
            example_sample = dataset[example_idx]
            if example_sample is None:
                continue

            # 处理样本对并生成描述
            result = process_sample_pair(
                flamingo,
                example_sample,
                query_sample,
                tokenizer,
                image_processor,
                device
            )

            if result:
                # 保存生成结果和相关信息
                sample_result = {
                    'query_image_id': query_sample['image_id'],
                    'example_image_id': example_sample['image_id'],
                    'similarity_score': float(similarity),
                    'ground_truth_captions': query_sample['captions'],
                    'generated_text': result['generated_text'],
                    'demo_caption': result['demo_caption'],
                    'query_filename': query_sample['file_name'],
                    'example_filename': example_sample['file_name']
                }
                generation_results.append(sample_result)

            progress_bar.update(1)

        count_num_samples += 1
        # 定期保存中间结果
        if count_num_samples % 1000 == 0:
            save_temp_results(generation_results, args.output_dir, count_num_samples)

    progress_bar.close()

    # 保存最终结果
    save_results(generation_results, args.output_dir)
    print("处理完成")

def save_temp_results(results, output_dir, num_samples):
    """保存临时结果"""
    temp_file = os.path.join(output_dir, f"generation_results_clustered_temp_{num_samples}.json")
    with open(temp_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"临时结果已保存至: {temp_file}")


def save_results(results, output_dir):
    """保存最终结果"""
    final_file = os.path.join(output_dir, "generation_results_clustered_5000.json")
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"最终结果已保存至: {final_file}")

if __name__ == "__main__":
    main()