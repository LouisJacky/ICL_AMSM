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
from RL_base.open_flamingo import create_model_and_transforms

import torch
from tqdm import tqdm
from collections import defaultdict
import re
from RL_base.open_flamingo.inference import ofv2_inference
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
                        default="/path/to/Code/OFv2_ICL_VQA/open_flamingo/eval/data/okvqa/texts_features.h5",
                        )
    parser.add_argument('--k_samples',
                        type=int,
                        default=40,
                        help='candidate samples')

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

def compute_similarity_batched(features, batch_size=100):
    num_samples = features.shape[0]
    sim_matrix = torch.zeros((num_samples, num_samples), device=features.device)

    for i in tqdm(range(0, num_samples, batch_size), desc="计算相似度"):
        batch_end = min(i + batch_size, num_samples)
        batch = features[i:batch_end]

        # 计算当前批次与所有样本的相似度
        sim = F.cosine_similarity(batch.unsqueeze(1), features.unsqueeze(0), dim=2)
        sim_matrix[i:batch_end] = sim

    return sim_matrix

def sample_and_label_policy_model(ofv2_model, train_dataloader, args, device, tokenizer, image_processor):
    ofv2_model.eval()
    all_results = defaultdict(list)
    dataset = train_dataloader.dataset  # 获取数据集对象

    # 加载保存的特征
    with h5py.File(args.features_file, 'r') as f:
        text_features = torch.from_numpy(f['text_features'][:]).to(device)
        texts = f['texts'][:]
        question_ids = f['question_ids'][:]

    # 计算相似度矩阵
    logging.info("计算样本间的相似度...")
    similarity_matrix = compute_similarity_batched(text_features)

    # 对每个查询样本找到最相似的k个样本
    k = args.k_samples
    total_samples = len(question_ids)

    with torch.no_grad():
        pbar = tqdm(total=total_samples, desc="Processing samples", unit="sample")

        for query_idx in range(total_samples):
            # 获取查询样本
            query_batch = dataset[query_idx]
            query_question_id = query_batch['question_id']

            # 获取相似度分数并找到top-k个最相似样本
            sim_scores = similarity_matrix[query_idx]
            sim_scores[query_idx] = -1  # 排除自身
            top_k_values, top_k_indices = torch.topk(sim_scores, k)

            # 处理每个相似样本
            for example_idx, similarity in zip(top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()):
                example_batch = dataset[example_idx]
                example_question_id = example_batch['question_id']

                # 计算置信度分数
                confidence_score = process_pair(
                    ofv2_model, query_batch, example_batch,
                    device, tokenizer, image_processor
                )

                # 保存结果
                pair_info = {
                    'query_question_id': int(query_question_id),
                    'example_question_id': int(example_question_id),
                    'confidence_score': confidence_score,
                    'similarity_score': float(similarity),
                }
                all_results['samples'].append(pair_info)

            pbar.update(1)
            if pbar.n % (total_samples // 10) == 0:
                save_temp_results(all_results, args.BENCHMARK)

        pbar.close()

    return all_results

def save_temp_results(results, benchmark):
    """保存临时结果"""
    temp_output_file = f"{benchmark}_label_similarity_temp.json"
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"临时结果已保存到 {temp_output_file}")

def get_word_confidence(ofv2_model, device, query_image, query_question, target_answer,
                        example_image, example_question, example_answer,
                        tokenizer, image_processor):
    debug = False
    prompt = f"<image>Question:{example_question} Short answer:{example_answer}.<|endofchunk|><image>Question:{query_question} Short answer:{target_answer}."

    target_tokens = tokenizer(f"{target_answer}", return_tensors="pt").input_ids.to(device)
    target_length = target_tokens.size(1)

    vision_x = [image_processor(example_image).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    vision_x = vision_x.to(device).half()

    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = ofv2_model(
            vision_x=vision_x,
            lang_x=prompt_tokens.input_ids,
            attention_mask=prompt_tokens.attention_mask,
            labels=None,
        )

        logits = outputs.logits[:, -target_length-2:-2, :]

        if debug:
            # 添加调试信息
            print("\nDebug Information:")
            print(f"Target answer: {target_answer}")
            print(f"Target tokens: {[tokenizer.decode([id.item()]) for id in target_tokens[0]]}")

        temperature = 1.0
        logits = logits / temperature
        scores = nn.functional.log_softmax(logits, dim=-1)
        probs = nn.functional.softmax(scores, dim=-1)

        target_probs = []
        for i in range(target_length):
            token_probs = probs[0, i]
            target_token_id = target_tokens[0, i].item()

            # 打印top-k预测
            if debug:
                top_probs, top_indices = token_probs.topk(5)
                print(f"\nPosition {i}:")
                print(f"Target token: {tokenizer.decode([target_token_id])}")
                print(f"Target probability: {token_probs[target_token_id].item():.4f}")
                print("Top 5 predictions:")
                for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    print(f"{tokenizer.decode([idx])}: {prob:.4f}")

            target_prob = max(token_probs[target_token_id].item(), 1e-10)
            target_probs.append(target_prob)

        avg_confidence = 0.0
        if len(target_probs) > 0:
            avg_confidence = sum(target_probs) / len(target_probs)

        if debug:
            print(f"\nFinal confidence score: {avg_confidence:.4f}")

    return avg_confidence

def process_pair(model, query_batch, example_batch, device, tokenizer, image_processor):
    """处理单个样本对，返回置信度分数"""
    confidence_score = get_word_confidence(
        model, device,
        query_batch['image'], query_batch['question'],
        query_batch['most_common_answer'],
        example_batch['image'], example_batch['question'],
        example_batch['most_common_answer'],
        tokenizer, image_processor
    )
    return confidence_score

def extract_and_save_features(ofv2_model, train_dataloader, args, device, clip_model, preprocess):
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

    with h5py.File(args.features_file, 'w') as f:
        f.create_dataset('text_features', data=all_text_features.numpy())
        f.create_dataset('texts', data=np.array(all_texts, dtype=h5py.special_dtype(vlen=str)))
        f.create_dataset('question_ids', data=np.array(all_question_ids))

        dt = h5py.special_dtype(vlen=str)
        id_to_idx = {str(qid): idx for idx, qid in enumerate(all_question_ids)}
        id_to_idx_group = f.create_group('id_to_idx')
        for qid, idx in id_to_idx.items():
            id_to_idx_group.attrs[qid] = idx

    print(f"特征已保存到 {args.features_file}")
    print(f"共保存了 {len(all_question_ids)} 个样本的特征")

# 主函数中的修改
if __name__ == "__main__":
    args = parse_args()
    output_file = f"{args.BENCHMARK}_label_confidence_{args.k_samples}.json"
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
    train_dataloader = get_okvqa_dataloader(
        args.okvqa_train_questions_json_path,
        args.okvqa_train_annotations_json_path,
        args.train_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # # 提取并保存特征
    # extract_and_save_features(ofv2_model, train_dataloader, args, device, clip_model, preprocess)

    # 运行标签预测
    labeled_samples = sample_and_label_policy_model(
        ofv2_model, train_dataloader, args, device, tokenizer, image_processor
    )

    # 保存结果
    # output_file = f"{args.BENCHMARK}_label_pair{args.k_samples}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_samples, f, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file}")