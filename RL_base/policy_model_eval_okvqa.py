import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import argparse
import random
import numpy as np
import UTILS as utils
import clip
from policy_model import AdaptiveMultiModalMatchingModel
import torch.nn as nn
import time
from tqdm import tqdm
from datetime import datetime

from label_vqa import get_okvqa_dataloader, OKVQADataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_feature_dim', type=int, default=512, help='Policy network final layer hidden state size.')
    parser.add_argument('--policy_num_heads', type=int, default=8,
                        help='Policy network')
    parser.add_argument('--policy_num_layers', type=int, default=2,
                        help='Policy network')
    parser.add_argument('--policy_dropout', type=float, default=0.0,
                        help='Policy network')

    parser.add_argument('--Modal', type=str, default='both', choices=['both', 'text', 'image'],
                        help='Modal used by policy_model: both/text/image')

    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='Policy network training batch size. Set to train_number by default.')

    parser.add_argument('--shot_number',
                        type=int,
                        default=8,
                        help='Prompt num')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--num_workers', type=int, default=4, help='random seed')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--base_dir', default='../datasets',
                        help='pascal base dir')

    parser.add_argument('--task', default='vqa', choices=['vqa', 'caption'])
    parser.add_argument('--BENCHMARK', default='okvqa', type=str,
                        help='dataset type:"vizwiz, okvqa')

    # 尝试调整学习率
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')

    parser.add_argument('--output_root', type=str, default='../log/ofv2_base/END_OUTPUT_linear')

    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    # parser.add_argument('--cuda', action='store_true', default=True,
    #                     help='enables CUDA training')

    parser.add_argument('--input_size', type=int, default=448)

    # parser.add_argument('--resume', type=bool, default=False,
    #                     help='Breakpoint recovery training')

    # parser.add_argument('--policy_model_checkpoint', type=str,
    #                     default="/path/to/AMSM/log/ofv2_base/END_OUTPUT_linear/okvqa/policy_model_epoch_2.pth",
    #                     help='Learned model checkpoint path on okvqa')

    parser.add_argument('--backbone', default="clip", type=str, help='backbone of retrival feature VIT、clip')

    parser.add_argument('--clip_type', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
                                 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
                        help='CLIP type')

    parser.add_argument('--train_VIT_Img_size', default=224, type=int, help='input size of VIT')

    parser.add_argument('--mode', default='train', type=str, help='train, val')
    parser.add_argument('--provide_ex_ans', type=bool, default=False, help='provide example answer')

    # ofv2
    # 添加OK-VQA相关的参数
    parser.add_argument(
        "--train_image_dir",
        type=str,
        help="Path to the okvqa annotations json file.",
        default="/path/to/datasets/ok_vqa/train2014",
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        help="Path to the okvqa annotations json file.",
        default="/path/to/datasets/ok_vqa/val2014",
    )

    parser.add_argument(
        "--okvqa_train_questions_json_path",
        type=str,
        help="Path to the okvqa questions json file.",
        default="/path/to/datasets/ok_vqa/OpenEnded_mscoco_train2014_questions.json",
    )
    parser.add_argument(
        "--okvqa_train_annotations_json_path",
        type=str,
        help="Path to the okvqa annotations json file.",
        default="/path/to/datasets/ok_vqa/mscoco_train2014_annotations.json",
    )
    parser.add_argument(
        "--okvqa_test_questions_json_path",
        type=str,
        help="Path to the okvqa questions json file.",
        default="/path/to/datasets/ok_vqa/OpenEnded_mscoco_val2014_questions.json",
    )
    parser.add_argument(
        "--okvqa_test_annotations_json_path",
        type=str,
        help="Path to the okvqa annotations json file.",
        default="/path/to/datasets/ok_vqa/mscoco_val2014_annotations.json",
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    # args.policy_model_checkpoint = os.path.join(args.output_path,f"policy_model_{args.Modal}_{args.clip_type}_epoch_2.pth")
    clip_type_name = args.clip_type.replace('/', '_')  # 将 'ViT-B/32' 转换为 'ViT-B_32'
    args.policy_model_checkpoint = os.path.join(args.output_path,
                                   f"policy_model_{args.Modal}_{clip_type_name}_epoch_2.pth")
    utils.create_dir(args.output_path)
    return args

def load_trained_policy_model(checkpoint_path, device, args):
    # 加载CLIP模型
    clip_model, preprocess = clip.load(args.clip_type, device=device)

    # 初始化策略模型
    # policy_model = MultiModalMatchingModel(clip_model, hidden_size=args.policy_feature_dim).to(device)
    policy_model = AdaptiveMultiModalMatchingModel(clip_model, hidden_size=args.policy_feature_dim, Modal=args.Modal).to(device)

    # 加载保存的模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 检查checkpoint的格式并相应地加载
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy_model.load_state_dict(checkpoint)

    # 将模型设置为评估模式
    policy_model.eval()

    return policy_model, preprocess


def evaluate_policy_model(policy_model, test_dataloader, train_dataloader, args, device, preprocess):
    policy_model.eval()
    # 创建question_id到样本的映射
    train_samples = {}
    for batch_idx, batch in enumerate(train_dataloader):
        for i in range(len(batch['question_id'])):
            train_samples[batch['question_id'][i]] = {
                'image': batch['image'][i],
                'question': batch['question'][i],
                'answers': batch['answers'][i],
                'most_common_answer': batch['most_common_answer'][i],
                'image_id': batch['image_id'][i],
                'question_id': batch['question_id'][i]
            }

    # 预计算所有训练样本的特征
    train_features = []
    train_question_ids = []  # 保存question_id的顺序
    with torch.no_grad():
        for train_batch in tqdm(train_dataloader, desc="预处理训练样本"):
            train_images = torch.stack([preprocess(img) for img in train_batch['image']]).to(device)
            if args.provide_ex_ans:
                example_qa_s = [f"{q} {a}" for q, a in
                                zip(train_batch['question'], train_batch['most_common_answer'])]
                batch_features = policy_model.encode_image_text(train_images, example_qa_s)
            else:
                train_questions = train_batch['question']
                batch_features = policy_model.encode_image_text(train_images, train_questions)
            train_features.append(batch_features)
            train_question_ids.extend(train_batch['question_id'])
    train_features = torch.cat(train_features, dim=0)

    results_dict = {}  # 用于存储每个测试样本的检索结果

    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc="处理测试样本"):
            test_images = torch.stack([preprocess(img) for img in test_batch['image']]).to(device)
            test_questions = test_batch['question']
            test_question_ids = test_batch['question_id']

            # 批量编码测试样本
            test_features = policy_model.encode_image_text(test_images, test_questions)

            # 计算相似度分数
            scores = policy_model.batch_compute_similarity(test_features, train_features)

            # 对每个测试样本找到最相似的k个训练样本
            k = args.shot_number
            topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)

            # 存储结果
            for i, test_question_id in enumerate(test_question_ids):
                top_examples = []
                for j, train_idx in enumerate(topk_indices[i]):
                    train_question_id = train_question_ids[train_idx]
                    train_sample = train_samples[train_question_id]

                    example_info = {
                        'score': float(topk_scores[i][j]),
                        'example_question_id': int(train_question_id),
                        'example_image_id': train_sample['image_id'],
                        'example_question': train_sample['question'],
                        'example_answers': train_sample['answers'],
                        'example_most_common_answer': train_sample['most_common_answer']
                    }
                    top_examples.append(example_info)

                results_dict[test_question_id] = {
                    'query_question_id': int(test_question_id),
                    'query_image_id': test_batch['image_id'][i],
                    'query_question': test_batch['question'][i],
                    'query_answers': test_batch['answers'][i],
                    'query_most_common_answer': test_batch['most_common_answer'][i],
                    'top_examples': top_examples
                }

    # 创建一个用于保存结果的目录
    results_dir = os.path.join(args.output_path, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    # 生成一个包含时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"best_examples_{timestamp}.json"

    # 保存结果
    output_file = os.path.join(results_dir, f'retrieval_results_{args.BENCHMARK}_{timestamp}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"检索结果已保存到: {output_file}")

    return results_dict

# 使用示例
if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    policy_model, preprocess = load_trained_policy_model(args.policy_model_checkpoint, device, args)

    # 加载数据集
    train_dataloader = get_okvqa_dataloader(
        args.okvqa_train_questions_json_path,
        args.okvqa_train_annotations_json_path,
        args.train_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="trn",
    )

    test_dataloader = get_okvqa_dataloader(
        args.okvqa_test_questions_json_path,
        args.okvqa_test_annotations_json_path,
        args.test_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val",
    )

    # 评估模型
    results = evaluate_policy_model(policy_model, test_dataloader, train_dataloader, args, device, preprocess)

    print("评估完成")
