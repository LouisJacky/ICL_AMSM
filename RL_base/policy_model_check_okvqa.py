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
import torch.nn as nn
import time
from tqdm import tqdm
from datetime import datetime
from open_flamingo import create_model_and_transforms

import torch
from tqdm import tqdm
from collections import defaultdict
import re
from open_flamingo.inference import ofv2_inference
from PIL import Image
import json
import logging
import time
from datetime import timedelta
import itertools
import h5py
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO)


class OKVQADataset(Dataset):
    def __init__(self, questions_file, annotations_file, image_dir, transform=None, split="trn"):
        # 加载问题和答案数据
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)['questions']
            # 创建question_id到question的映射
            self.id_to_question = {q['question_id']: q for q in questions_data}
            self.question_ids = [q['question_id'] for q in questions_data]  # 保存所有question_id

        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)['annotations']
            # 创建question_id到annotation的映射
            self.id_to_annotation = {ann['question_id']: ann for ann in annotations_data}

        self.image_dir = image_dir

        self.split = split

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        question_id = self.question_ids[idx]
        question_data = self.id_to_question[question_id]
        annotation = self.id_to_annotation[question_id]

        # 加载图像
        image_id = question_data['image_id']
        # OK-VQA使用COCO格式的图像名称
        if self.split == "trn":
            image_path = os.path.join(self.image_dir, f'COCO_train2014_{str(image_id).zfill(12)}.jpg')
        elif self.split == "val":
            image_path = os.path.join(self.image_dir, f'COCO_val2014_{str(image_id).zfill(12)}.jpg')
        image = Image.open(image_path).convert('RGB')

        # 获取问题和答案
        question = question_data['question']
        answers = [ans['answer'] for ans in annotation['answers']]

        # 找出最常见的答案作为标准答案
        answer_counts = {}
        for ans in answers:
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        most_common_answer = max(answer_counts.items(), key=lambda x: x[1])[0]

        return {
            'image': image,
            'question': question,
            'answers': answers,
            'most_common_answer': most_common_answer,
            'question_id': question_id,
            'image_id': image_id
        }


def custom_collate(batch):
    # collate函数保持不变
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answers'] for item in batch]
    most_common_answers = [item['most_common_answer'] for item in batch]
    question_ids = [item['question_id'] for item in batch]
    image_ids = [item['image_id'] for item in batch]

    return {
        'image': images,
        'question': questions,
        'answers': answers,
        'most_common_answer': most_common_answers,
        'question_id': question_ids,
        'image_id': image_ids
    }


def get_okvqa_dataloader(questions_file, annotations_file, image_dir, batch_size=32, shuffle=True, num_workers=4, split="trn"):
    dataset = OKVQADataset(questions_file, annotations_file, image_dir, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=custom_collate)

def parse_args():
    parser = argparse.ArgumentParser()
    # ofv2
    parser.add_argument('--lm_path', type=str, default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str, default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str, default="/path/to/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='3')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--BENCHMARK', default='okvqa', type=str,
                        help='dataset type:"vizwiz, okvqa"')
    parser.add_argument('--num_workers', default=2, type=int)

    # ## VizWiz Dataset
    # parser.add_argument(
    #     "--train_image_dir",
    #     type=str,
    #     help="Path to the vizwiz annotations json file.",
    #     default="/path/to/datasets/VizWiz/train",
    # )
    # parser.add_argument(
    #     "--test_image_dir",
    #     type=str,
    #     help="Path to the vizwiz annotations json file.",
    #     default="/path/to/datasets/VizWiz/val",
    # )
    #
    # parser.add_argument(
    #     "--vizwiz_train_questions_json_path",
    #     type=str,
    #     help="Path to the vizwiz questions json file.",
    #     default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/clustered_train_questions_vqa_format.json",
    #     # default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/train_questions_vqa_format.json",
    # )
    # parser.add_argument(
    #     "--vizwiz_train_annotations_json_path",
    #     type=str,
    #     help="Path to the vizwiz annotations json file.",
    #     default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/sampled/sampled_train_annotations_vqa_format.json",
    #     # default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/train_annotations_vqa_format.json",
    # )
    # parser.add_argument(
    #     "--vizwiz_test_questions_json_path",
    #     type=str,
    #     help="Path to the vizwiz questions json file.",
    #     default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/clustered_val_questions_vqa_format.json",
    # )
    # parser.add_argument(
    #     "--vizwiz_test_annotations_json_path",
    #     type=str,
    #     help="Path to the vizwiz annotations json file.",
    #     default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/sampled/sampled_val_annotations_vqa_format.json",
    # )
    ## okvqa Dataset
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

    parser.add_argument('--features_file', type=str,
                        help="Path to the okvqa features_file.",
                        default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/okvqa/texts_features",
                        )
    # parser.add_argument('--k_samples',
    #                     type=int,
    #                     default=40,
    #                     help='candidate samples')

    args = parser.parse_args()
    return args


def extract_short_answer(text):
    pattern = r'Short answer:\s*(.*?)\.'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    else:
        pattern = r'Answer:\s*(.*?)\.'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def save_temp_results(results, benchmark):
    """保存临时结果"""
    temp_output_file = f"{benchmark}_label_similarity_temp.json"
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"临时结果已保存到 {temp_output_file}")


def extract_and_save_features(ofv2_model, train_dataloader, args, device, clip_model, preprocess, split="trn"):
    ofv2_model.eval()

    all_text_features = []
    all_texts = []
    all_question_ids = []

    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="提取特征"):
            combined_texts = [f"{q} {a}" for q, a in zip(batch['question'], batch['most_common_answer'])]
            text_features = clip_model.encode_text(clip.tokenize(combined_texts).to(device)).float()
            all_text_features.append(text_features.cpu())
            all_texts.extend(combined_texts)
            all_question_ids.extend(batch['question_id'])

    all_text_features = torch.cat(all_text_features, dim=0)
    output_features_file = ""
    if split == "trn":
        output_features_file = args.features_file + "_trn.h5"
    elif split == "val":
        output_features_file = args.features_file + "_val.h5"
    with h5py.File(output_features_file, 'w') as f:
        f.create_dataset('text_features', data=all_text_features.numpy())
        f.create_dataset('texts', data=np.array(all_texts, dtype=h5py.special_dtype(vlen=str)))
        f.create_dataset('question_ids', data=np.array(all_question_ids))

        dt = h5py.special_dtype(vlen=str)
        id_to_idx = {str(qid): idx for idx, qid in enumerate(all_question_ids)}
        id_to_idx_group = f.create_group('id_to_idx')
        for qid, idx in id_to_idx.items():
            id_to_idx_group.attrs[qid] = idx

    print(f"特征已保存到 {output_features_file}")
    print(f"共保存了 {len(all_question_ids)} 个样本的特征")


def match_test_train_samples(args, device):
    print("开始加载特征文件...")
    # 加载训练集和测试集特征
    with h5py.File(args.features_file + "_trn.h5", 'r') as f:
        train_features = torch.from_numpy(f['text_features'][:]).to(device)
        train_question_ids = f['question_ids'][:]
    print(f"已加载训练集特征: {len(train_question_ids)} 个样本")

    with h5py.File(args.features_file + "_val.h5", 'r') as f:
        test_features = torch.from_numpy(f['text_features'][:]).to(device)
        test_question_ids = f['question_ids'][:]
    print(f"已加载测试集特征: {len(test_question_ids)} 个样本")

    print("加载数据集...")
    # 加载训练集数据以获取完整信息
    train_dataset = OKVQADataset(
        args.okvqa_train_questions_json_path,
        args.okvqa_train_annotations_json_path,
        args.train_image_dir,
        split="trn"
    )

    print("处理训练集数据...")
    train_samples = {}
    for qid in tqdm(train_dataset.question_ids, desc="处理训练样本"):
        train_samples[qid] = train_dataset.id_to_question[qid] | train_dataset.id_to_annotation[qid]

    print("加载测试集数据...")
    # 加载测试集数据
    test_dataset = OKVQADataset(
        args.okvqa_test_questions_json_path,
        args.okvqa_test_annotations_json_path,
        args.test_image_dir,
        split="val"
    )

    results_dict = {}
    batch_size = 100
    k = 4  # 每个测试样本匹配4个训练样本

    total_batches = (len(test_features) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="计算相似度匹配")

    # 分批计算相似度以节省内存
    for i in range(0, len(test_features), batch_size):
        batch_end = min(i + batch_size, len(test_features))
        test_batch = test_features[i:batch_end]

        # 计算当前测试批次与所有训练样本的相似度
        sim = F.cosine_similarity(test_batch.unsqueeze(1), train_features.unsqueeze(0), dim=2)

        # 对每个测试样本找到top-k个训练样本
        topk_scores, topk_indices = torch.topk(sim, k=k, dim=1)

        # 保存匹配结果
        for j, (scores, indices) in enumerate(zip(topk_scores, topk_indices)):
            test_idx = i + j
            test_qid = int(test_question_ids[test_idx])
            test_data = test_dataset.id_to_question[test_qid]
            test_anno = test_dataset.id_to_annotation[test_qid]

            top_examples = []
            for score, train_idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
                train_qid = int(train_question_ids[train_idx])
                train_data = train_samples[train_qid]

                example_info = {
                    'score': float(score),
                    'example_question_id': train_qid,
                    'example_image_id': train_data['image_id'],
                    'example_question': train_data['question'],
                    'example_answers': [ans['answer'] for ans in train_data['answers']],
                    'example_most_common_answer': max(
                        ((ans['answer'], sum(1 for a in train_data['answers'] if a['answer'] == ans['answer']))
                         for ans in train_data['answers']),
                        key=lambda x: x[1]
                    )[0]
                }
                top_examples.append(example_info)

            results_dict[test_qid] = {
                'query_question_id': test_qid,
                'query_image_id': test_data['image_id'],
                'query_question': test_data['question'],
                'query_answers': [ans['answer'] for ans in test_anno['answers']],
                'query_most_common_answer': max(
                    ((ans['answer'], sum(1 for a in test_anno['answers'] if a['answer'] == ans['answer']))
                     for ans in test_anno['answers']),
                    key=lambda x: x[1]
                )[0],
                'top_examples': top_examples
            }

        # 更新进度条
        progress_bar.update(1)

        # 定期保存临时结果
        if (i // batch_size) % 10 == 0:
            save_temp_results(results_dict, f"{args.BENCHMARK}_test_train_matching")
            progress_bar.set_postfix({'已保存样本数': len(results_dict)})

    progress_bar.close()
    print(f"匹配完成，共处理 {len(results_dict)} 个测试样本")
    return results_dict

# 主函数中的修改
if __name__ == "__main__":
    args = parse_args()
    output_file = f"{args.BENCHMARK}_test_train_matching.json"
    print(output_file)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # 加载模型和处理器
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

    # 加载训练数据
    # train_dataloader = get_okvqa_dataloader(
    #     args.okvqa_train_questions_json_path,
    #     args.okvqa_train_annotations_json_path,
    #     args.train_image_dir,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers
    # )

    test_dataloader = get_okvqa_dataloader(
        args.okvqa_test_questions_json_path,
        args.okvqa_test_annotations_json_path,
        args.test_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="val",
    )

    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # # 提取并保存特征
    # extract_and_save_features(ofv2_model, train_dataloader, args, device, clip_model, preprocess, split="trn")
    extract_and_save_features(ofv2_model, test_dataloader, args, device, clip_model, preprocess, split="val")

    # 添加新的匹配过程
    matched_samples = match_test_train_samples(args, device)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_samples, f, ensure_ascii=False, indent=4)

    print(f"匹配结果已保存到 {output_file}")