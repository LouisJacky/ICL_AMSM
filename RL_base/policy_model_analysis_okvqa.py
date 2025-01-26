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
from policy_model import ResMultiModalMatchingModel
from policy_model_eval_okvqa import load_trained_policy_model, get_okvqa_dataloader
import h5py
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--shot_number', type=int, default=8)

    # 数据路径
    parser.add_argument('--train_image_dir', type=str, default="/data16tb/ljq/datasets/ok_vqa/train2014")
    parser.add_argument('--test_image_dir', type=str, default="/data16tb/ljq/datasets/ok_vqa/val2014")
    parser.add_argument('--okvqa_train_questions_json_path', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument('--okvqa_train_annotations_json_path', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/mscoco_train2014_annotations.json")
    parser.add_argument('--okvqa_test_questions_json_path', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument('--okvqa_test_annotations_json_path', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/mscoco_val2014_annotations.json")

    # 模型相关
    parser.add_argument('--policy_feature_dim', type=int, default=512)
    parser.add_argument('--policy_model_checkpoint', type=str,
                        default="/data16tb/ljq/Code/ICL_diversity_ofv3/log/ofv2_base/END_OUTPUT_linear/okvqa/policy_model_both_RN50x16_epoch_2.pth")
    parser.add_argument('--features_file', type=str,
                        default="/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/okvqa/texts_features")
    parser.add_argument('--clip_type', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
                                 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
                        help='CLIP type')
    parser.add_argument('--Modal', type=str, default='both', choices=['both', 'text', 'image'],
                        help='Modal used by policy_model: both/text/image')
    args = parser.parse_args()
    return args


def save_method_results(results, method_name, args):
    """保存单个方法的结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理结果确保可序列化
    def check_json_serializable(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [check_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): check_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        return obj
    
    output_data = {
        'metadata': {
            'method': method_name,
            'sample_size': args.sample_size,
            'shot_number': args.shot_number,
            'timestamp': timestamp
        },
        'results': check_json_serializable(results)
    }
    
    output_file = os.path.join(output_dir, f'{method_name}_matching_{timestamp}.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"{method_name}方法的结果已保存到: {output_file}")
    except TypeError as e:
        print(f"保存{method_name}结果时发生序列化错误: {e}")
        print("结果数据类型:")
        print({k: type(v) for k, v in output_data.items()})

def decode_bytes(obj):
    """递归地将bytes对象转换为字符串"""
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, list):
        return [decode_bytes(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: decode_bytes(value) for key, value in obj.items()}
    return obj

def get_multimodal_matches(policy_model, test_samples, train_dataloader, args, device, preprocess):
    """使用多模态策略模型进行匹配"""
    results = {}
    train_samples = {}

    # 预处理训练数据
    for batch in train_dataloader:
        for i in range(len(batch['question_id'])):
            train_samples[batch['question_id'][i]] = {
                'image': batch['image'][i],
                'question': batch['question'][i],
                'answers': batch['answers'][i],
                'most_common_answer': batch['most_common_answer'][i],
                'image_id': batch['image_id'][i]
            }

    # 计算所有训练样本的特征
    train_features = []
    train_question_ids = []
    with torch.no_grad():
        for train_batch in tqdm(train_dataloader, desc="计算训练样本特征"):
            train_images = torch.stack([preprocess(img) for img in train_batch['image']]).to(device)
            train_questions = train_batch['question']
            batch_features = policy_model.encode_image_text(train_images, train_questions)
            train_features.append(batch_features)
            train_question_ids.extend(train_batch['question_id'])
    train_features = torch.cat(train_features, dim=0)

    # 对测试样本进行匹配
    with torch.no_grad():
        for test_sample in tqdm(test_samples, desc="多模态匹配"):
            test_image = preprocess(test_sample['image']).unsqueeze(0).to(device)
            test_question = [test_sample['question']]
            test_features = policy_model.encode_image_text(test_image, test_question)

            # 计算相似度并获取top-k
            scores = policy_model.batch_compute_similarity(test_features, train_features)
            topk_scores, topk_indices = torch.topk(scores, k=args.shot_number, dim=1)

            # 保存结果
            top_examples = []
            for j, train_idx in enumerate(topk_indices[0]):
                train_qid = train_question_ids[train_idx]
                train_sample = train_samples[train_qid]
                example_info = {
                    'score': float(topk_scores[0][j]),
                    'example_question': train_sample['question'],
                    'example_answers': train_sample['answers'],
                    'example_most_common_answer': train_sample['most_common_answer'],
                    'example_image_id': train_sample['image_id']
                }
                top_examples.append(example_info)

            results[test_sample['question_id']] = {
                'method': 'multimodal',
                'top_examples': top_examples
            }

    # 确保返回的数据是JSON可序列化的
    for qid in results:
        results[qid] = decode_bytes(results[qid])
        if 'top_examples' in results[qid]:
            for example in results[qid]['top_examples']:
                example['score'] = float(example['score'])
    
    # 保存多模态匹配结果
    save_method_results(results, 'multimodal', args)
    return results


# def get_text_matches(test_samples, args, device):
#     """使用文本特征进行匹配"""
#     results = {}
#
#     # 加载特征文件
#     with h5py.File(args.features_file + "_trn.h5", 'r') as f:
#         train_features = torch.from_numpy(f['text_features'][:]).to(device)
#         train_texts = f['texts'][:]
#         train_question_ids = f['question_ids'][:]
#
#     # 加载训练集数据以获取完整信息
#     train_dataset = get_okvqa_dataloader(
#         args.okvqa_train_questions_json_path,
#         args.okvqa_train_annotations_json_path,
#         args.train_image_dir,
#         batch_size=1,
#         num_workers=args.num_workers,
#         split="trn"
#     ).dataset
#
#     # 创建question_id到完整信息的映射
#     train_info = {}
#     for qid in train_question_ids:
#         qid = int(qid)
#         if qid in train_dataset.id_to_question:
#             question_data = train_dataset.id_to_question[qid]
#             annotation_data = train_dataset.id_to_annotation[qid]
#             train_info[qid] = {
#                 'question': question_data['question'],
#                 'answers': [ans['answer'] for ans in annotation_data['answers']],
#                 'image_id': question_data['image_id']
#             }
#             # 找出最常见的答案
#             answer_counts = {}
#             for ans in train_info[qid]['answers']:
#                 answer_counts[ans] = answer_counts.get(ans, 0) + 1
#             train_info[qid]['most_common_answer'] = max(answer_counts.items(), key=lambda x: x[1])[0]
#
#     with h5py.File(args.features_file + "_val.h5", 'r') as f:
#         test_features = torch.from_numpy(f['text_features'][:]).to(device)
#         test_texts = f['texts'][:]
#         test_question_ids = f['question_ids'][:]
#
#     # 为每个测试样本找到匹配
#     for test_sample in tqdm(test_samples, desc="文本匹配"):
#         test_idx = np.where(test_question_ids == test_sample['question_id'])[0][0]
#         test_feature = test_features[test_idx].unsqueeze(0)
#
#         # 计算相似度
#         sim = F.cosine_similarity(test_feature, train_features, dim=1)
#         topk_scores, topk_indices = torch.topk(sim, k=args.shot_number)
#
#         # 保存结果
#         top_examples = []
#         for score, idx in zip(topk_scores, topk_indices):
#             train_qid = int(train_question_ids[idx])
#             if train_qid in train_info:
#                 example_info = {
#                     'score': float(score),
#                     'example_question': train_info[train_qid]['question'],
#                     'example_answers': train_info[train_qid]['answers'],
#                     'example_most_common_answer': train_info[train_qid]['most_common_answer'],
#                     'example_image_id': train_info[train_qid]['image_id']
#                 }
#                 top_examples.append(example_info)
#
#         results[test_sample['question_id']] = {
#             'method': 'text',
#             'top_examples': top_examples
#         }
#
#     # 确保数据可序列化并保存结果
#     results = decode_bytes(results)
#     save_method_results(results, 'text', args)
#     return results

def get_text_matches(clip_model, test_samples, train_dataloader, args, device):
    """使用CLIP文本特征进行匹配"""
    results = {}
    train_samples = {}

    # 计算所有训练样本的文本特征
    train_features = []
    train_info = []

    print("计算训练样本的文本特征...")
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="处理训练文本"):
            # 处理文本批次
            train_texts = clip.tokenize(batch['question']).to(device)
            text_features = clip_model.encode_text(train_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            train_features.append(text_features)
            for i in range(len(batch['question_id'])):
                train_info.append({
                    'question': batch['question'][i],
                    'answers': batch['answers'][i],
                    'most_common_answer': batch['most_common_answer'][i],
                    'image_id': batch['image_id'][i]
                })

    train_features = torch.cat(train_features, dim=0)

    # 对每个测试样本进行匹配
    print("开始文本特征匹配...")
    with torch.no_grad():
        for test_sample in tqdm(test_samples, desc="文本匹配"):
            # 处理测试文本
            test_text = clip.tokenize([test_sample['question']]).to(device)
            test_features = clip_model.encode_text(test_text)
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
                    'example_question': train_sample['question'],
                    'example_answers': train_sample['answers'],
                    'example_most_common_answer': train_sample['most_common_answer'],
                    'example_image_id': train_sample['image_id']
                }
                top_examples.append(example_info)

            results[test_sample['question_id']] = {
                'method': 'text',
                'top_examples': top_examples
            }

    # 确保返回的数据是JSON可序列化的
    for qid in results:
        results[qid] = decode_bytes(results[qid])
        if 'top_examples' in results[qid]:
            for example in results[qid]['top_examples']:
                example['score'] = float(example['score'])
                if 'example_image_id' in example:
                    example['example_image_id'] = int(example['example_image_id'])

    # 保存文本匹配结果
    save_method_results(results, 'text', args)
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
            # 处理图像批次
            train_images = torch.stack([preprocess(img) for img in batch['image']]).to(device)
            image_features = clip_model.encode_image(train_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            train_features.append(image_features)
            for i in range(len(batch['question_id'])):
                train_info.append({
                    'question': batch['question'][i],
                    'answers': batch['answers'][i],
                    'most_common_answer': batch['most_common_answer'][i],
                    'image_id': batch['image_id'][i]
                })

    train_features = torch.cat(train_features, dim=0)

    # 对每个测试样本进行匹配
    print("开始图像特征匹配...")
    with torch.no_grad():
        for test_sample in tqdm(test_samples, desc="图像匹配"):
            # 处理测试图像
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
                    'example_question': train_sample['question'],
                    'example_answers': train_sample['answers'],
                    'example_most_common_answer': train_sample['most_common_answer'],
                    'example_image_id': train_sample['image_id']
                }
                top_examples.append(example_info)

            results[test_sample['question_id']] = {
                'method': 'image',
                'top_examples': top_examples
            }

    # 确保返回的数据是JSON可序列化的
    for qid in results:
        results[qid] = decode_bytes(results[qid])
        if 'top_examples' in results[qid]:
            for example in results[qid]['top_examples']:
                example['score'] = float(example['score'])
                if 'example_image_id' in example:
                    example['example_image_id'] = int(example['example_image_id'])

    # 保存图像匹配结果
    save_method_results(results, 'image', args)
    return results


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("加载CLIP模型")
    # 加载CLIP模型
    clip_model, preprocess = clip.load(args.clip_type, device=device)
    clip_model.eval()  # 设置为评估模式

    print("加载数据集")
    # 加载数据加载器
    test_dataloader = get_okvqa_dataloader(
        args.okvqa_test_questions_json_path,
        args.okvqa_test_annotations_json_path,
        args.test_image_dir,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        split="val"
    )

    train_dataloader = get_okvqa_dataloader(
        args.okvqa_train_questions_json_path,
        args.okvqa_train_annotations_json_path,
        args.train_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="trn"
    )

    print("采样测试样本")
    # 获取数据集总长度
    total_samples = len(test_dataloader.dataset)
    if args.sample_size is not None:
        # 生成随机索引
        random_indices = torch.randperm(total_samples)[:args.sample_size]
    else:
        # 当sample_size为None时，使用所有样本的顺序索引
        random_indices = torch.arange(total_samples)

    test_samples = []
    for idx in random_indices:
        batch = test_dataloader.dataset[idx.item()]
        sample = {k: v if isinstance(v, (list, torch.Tensor)) else v
                  for k, v in batch.items()}
        test_samples.append(sample)


    try:
        # 获取并保存多模态匹配结果
        print("开始多模态匹配...")
        # 加载策略模型
        policy_model, _ = load_trained_policy_model(args.policy_model_checkpoint, device, args)
        multimodal_results = get_multimodal_matches(policy_model, test_samples, train_dataloader, args, device, preprocess)
        
        # 获取并保存文本匹配结果
        print("开始文本匹配...")
        # text_results = get_text_matches(test_samples, args, device)
        text_results = get_text_matches(clip_model, test_samples, train_dataloader, args, device)

        # 获取并保存图像匹配结果
        print("开始图像匹配...")
        image_results = get_image_matches(clip_model, test_samples, train_dataloader, args, device, preprocess)
        
        print("所有分析完成！")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()