import json
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from PIL import Image
# from RL_base.UTILS import process_text_embedding, process_image_embedding
import argparse
import random
import numpy as np
import UTILS as utils
import clip
from policy_model import AdaptiveMultiModalMatchingModel
import torch.nn as nn
import time
import torch.nn.functional as F
import math  # 添加这行来导入math模块
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

class VQADataset(Dataset):
    def __init__(self, json_file, image_dir, num_pairs=20, transform=None):
        with open(json_file, 'r') as f:
            raw_data = json.load(f)

        # 重新组织数据结构
        self.data = []
        for item in raw_data:
            query_info = {
                'query_image': item['query_image'],
                'query_question': item['query_question'],
                'query_answers': item['query_answers'],
                'examples': item['examples']  # 所有可能的示例
            }
            self.data.append(query_info)

        self.image_dir = image_dir
        self.num_pairs = num_pairs  # 每个query要生成的样本对数量

        # 为每个query创建一个示例池，记录已使用的示例索引
        self.example_pools = []
        for item in self.data:
            total_examples = len(item['examples'])
            self.example_pools.append({
                'total': total_examples,
                'used': set(),  # 记录已使用的示例索引
                'current': []  # 当前epoch使用的示例索引
            })
        self.current_epoch = 0

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch):
        """设置当前epoch并为每个query重新选择示例"""
        self.current_epoch = epoch

        # 为每个query重新选择示例
        for pool in self.example_pools:
            total_examples = pool['total']
            used_examples = pool['used']

            # 如果剩余未使用的示例不足，重置已使用集合
            if total_examples - len(used_examples) < self.num_pairs:
                used_examples.clear()

            # 从未使用的示例中选择
            available = list(set(range(total_examples)) - used_examples)
            selected = random.sample(available, min(self.num_pairs, len(available)))

            # 更新已使用和当前使用的示例
            used_examples.update(selected)
            pool['current'] = selected

    def __getitem__(self, idx):
        item = self.data[idx]

        # # 为当前query随机选择num_pairs个示例
        # selected_examples = random.sample(item['examples'], min(self.num_pairs, len(item['examples'])))
        # 获取此query的所有可用示例
        all_examples = item['examples']

        # 使用当前epoch为该query预选的示例
        selected_indices = self.example_pools[idx]['current']
        selected_examples = [all_examples[i] for i in selected_indices]

        # 构建样本对
        query_images = []
        example_images = []
        query_questions = []
        example_questions = []
        query_answers = []
        example_answers = []
        labels = []
        example_most_common_answers = []

        # 加载query图像(只需加载一次)
        query_image = Image.open(f"{self.image_dir}/{item['query_image']}")

        for example in selected_examples:
            # 加载示例图像
            example_image = Image.open(f"{self.image_dir}/{example['example_image']}")

            query_images.append(query_image)
            example_images.append(example_image)
            query_questions.append(item['query_question'])
            example_questions.append(example['example_question'])
            query_answers.append(item['query_answers'])
            example_answers.append(example['example_answers'])
            labels.append(example['label'])

            # 找出最常见的答案作为标准答案
            example_answer_counts = {}
            for ans in example['example_answers']:
                example_answer_counts[ans] = example_answer_counts.get(ans, 0) + 1
            example_most_common_answer = max(example_answer_counts.items(), key=lambda x: x[1])[0]
            example_most_common_answers.append(example_most_common_answer)

        return {
            'query_image': query_images,
            'example_image': example_images,
            'query_question': query_questions,
            'example_question': example_questions,
            'query_answers': query_answers,
            'example_answers': example_answers,
            'label': torch.tensor(labels, dtype=torch.float32),
            'example_most_common_answers': example_most_common_answers,
        }


def custom_collate(batch):
    # 展平所有样本对
    query_images = [img for item in batch for img in item['query_image']]
    example_images = [img for item in batch for img in item['example_image']]
    query_questions = [q for item in batch for q in item['query_question']]
    example_questions = [q for item in batch for q in item['example_question']]
    query_answers = [ans for item in batch for ans in item['query_answers']]
    example_answers = [ans for item in batch for ans in item['example_answers']]
    labels = torch.cat([item['label'] for item in batch])
    example_most_common_answers = [ans for item in batch for ans in item['example_most_common_answers']]

    return {
        'query_image': query_images,
        'example_image': example_images,
        'query_question': query_questions,
        'example_question': example_questions,
        'query_answers': query_answers,
        'example_answers': example_answers,
        'label': labels,
        'example_most_common_answers': example_most_common_answers  # 添加这一行
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_feature_dim', type=int, default=512, help='Policy network final layer hidden state size.')
    parser.add_argument('--Modal', type=str, default='both', choices=['both', 'text', 'image'],
                        help='Modal used by policy_model: both/text/image')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Policy network training batch size. Set to train_number by default.')

    parser.add_argument('--shot_number',
                        type=int,
                        default=4,
                        help='Prompt num')

    parser.add_argument('--candidate_num', default=32, type=int, help='candidate examples num')

    parser.add_argument('--select_number',
                        type=int,
                        default=32,
                        help='candidate num')

    parser.add_argument('--seed', type=int, default=3, help='random seed')

    parser.add_argument('--num_workers', type=int, default=4, help='random seed')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--base_dir', default='../datasets',
                        help='pascal base dir')

    parser.add_argument('--task', default='vqa', choices=['vqa', 'caption'])
    parser.add_argument('--BENCHMARK', default='okvqa', type=str,
                        help='dataset type:"vizwiz, okvqa')
    # 尝试调整学习率
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')

    parser.add_argument('--output_root', type=str, default='../log/ofv2_base/END_OUTPUT_linear')

    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    # parser.add_argument('--cuda', action='store_true', default=True,
    #                     help='enables CUDA training')

    parser.add_argument('--input_size', type=int, default=448)

    parser.add_argument('--resume', type=bool, default=False,
                        help='Breakpoint recovery training')
    parser.add_argument('--resume_checkpoint', type=str,
                        default="/data16tb/ljq/Code/ICL_diversity_ofv3/log/ofv2_base/END_OUTPUT_linear/okvqa/policy_model_epoch_3.pth",
                        help='Learned model checkpoint path')

    parser.add_argument('--backbone', default="clip", type=str, help='backbone of retrival feature VIT、clip')

    # 添加新的参数用于配置CLIP版本
    parser.add_argument('--clip_type', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
                                 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
                        help='CLIP type')

    parser.add_argument('--train_VIT_Img_size', default=224, type=int, help='input size of VIT')

    parser.add_argument('--mode', default='train', type=str, help='train, val')
    parser.add_argument('--provide_ex_ans', type=bool, default=False, help='provide example answer')

    # ofv2
    ## VizWiz_labeled Dataset
    parser.add_argument(
        "--json_file",
        type=str,
        help="Path to the paired dataset json file",
        default='/data16tb/ljq/Code/ICL_diversity_ofv3/data/okvqa_labeled_confidence.json',
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the COCO train2014 images directory",
        default="/data16tb/ljq/datasets/ok_vqa/train2014",
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    utils.create_dir(args.output_path)
    return args

def preprocess_batch(images, preprocess_fn):
    return torch.stack([preprocess_fn(image) for image in images])

if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU

    # 使用示例

    # 创建数据集和数据加载器
    # 创建数据集
    dataset = VQADataset(args.json_file, args.image_dir, num_pairs=args.candidate_num)

    # 创建DataLoader
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=custom_collate)

    # 加载图像和文本编码器
    clip_model, image_preprocess = clip.load(args.clip_type, device=device)

    policy_model = AdaptiveMultiModalMatchingModel(
        clip_model,
        hidden_size=args.policy_feature_dim,
        Modal=args.Modal
    ).to(device)


    def custom_loss(outputs, labels, num_pairs):
        num_query = len(outputs) // num_pairs
        total_loss = 0.0
        # temperature = 0.5  # 可以调整这个温度参数

        for i in range(num_query):
            # 获取当前query的所有样本对的输出和标签
            start_idx = i * num_pairs
            end_idx = (i + 1) * num_pairs
            query_outputs = outputs[start_idx:end_idx]
            query_labels = labels[start_idx:end_idx]

            # 计算平均标签
            mean_label = torch.mean(query_labels)

            # 计算高于平均值和低于平均值的样本输出总和
            above_mean_mask = query_labels >= mean_label
            below_mean_mask = query_labels <= mean_label

            above_mean_mean = (query_outputs[above_mean_mask]).mean()
            below_mean_mean = (query_outputs[below_mean_mask]).mean()

            # 打印差值
            difference = above_mean_mean - below_mean_mean
            print(f"高均值: {above_mean_mean:.4f}, "
                  f"低均值: {below_mean_mean:.4f}, "
                  f"差值: {difference:.4f}, "
                  # f"学习率: {new_lr:.4f}"
                  )

            # 第二层：对样本对进行两两组合
            pair_losses = []
            for j in range(num_pairs):
                for k in range(j + 1, num_pairs):
                    # 获取当前对的输出和标签
                    pair_outputs = torch.stack([query_outputs[j], query_outputs[k]])
                    pair_labels = torch.stack([query_labels[j], query_labels[k]])

                    # 计算这对样本的平均标签
                    pair_mean_label = torch.mean(pair_labels)
                    pair_above_mean_mask = pair_labels >= pair_mean_label

                    # 计算这对样本的softmax概率
                    pair_output_probs = torch.nn.functional.softmax(pair_outputs, dim=-1)

                    # 计算这对样本的损失
                    pair_loss = (pair_mean_label - pair_labels) * (
                                torch.log(pair_output_probs))  # 因为是两个样本，所以用log(2)
                    pair_losses.append(torch.mean(pair_loss[pair_above_mean_mask]))

            sample_losses = torch.mean(torch.stack(pair_losses))

            # 累加损失
            total_loss += torch.sum(sample_losses)

        return total_loss / num_query


    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)
    scheduler = None

    # 加载checkpoint (如果存在)
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume_checkpoint
        if os.path.isfile(checkpoint_path):
            print(f"加载checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            policy_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从epoch {start_epoch}继续训练")
        else:
            print(f"未找到checkpoint: {checkpoint_path}")

    # # 将模型移动到指定设备
    # policy_model = policy_model.to(device)

    # 在训练循环开始前添加
    start_time = time.time()
    total_batches_processed = 0
    total_batch_time = 0

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        policy_model.train()
        total_batches = len(dataloader)
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            # 处理输入数据
            query_images = preprocess_batch(batch['query_image'], image_preprocess).to(device)
            example_images = preprocess_batch(batch['example_image'], image_preprocess).to(device)
            query_questions = batch['query_question']
            labels = batch['label'].to(device)

            if args.provide_ex_ans:
                # 拼接示例问题和答案
                example_qa_s = [f"{q} {a}" for q, a in
                                     zip(batch['example_question'], batch['example_most_common_answers'])]
                # 前向传播
                outputs = policy_model(query_images, query_questions, example_images, example_qa_s)

            else:
                example_questions = batch['example_question']
                # 前向传播
                outputs = policy_model(query_images, query_questions, example_images, example_questions)

            # 使用自定义损失函数计算损失
            loss = custom_loss(outputs, labels, args.candidate_num)

            # 在训练循环中添加调试信息
            # print(f"输出形状: {outputs.shape}, 标签形状: {labels.shape}")
            print(f"输出样本: {outputs[:min(5, len(outputs))]}")
            print(f"标签样本: {labels[:min(5, len(labels))]}")

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 更新进度信息
            batch_time = time.time() - batch_start_time
            total_batch_time += batch_time
            total_batches_processed += 1
            avg_batch_time = total_batch_time / total_batches_processed

            # 计算剩余时间
            batches_left = total_batches - (batch_idx + 1)
            time_left = batches_left * avg_batch_time
            epochs_left = args.epochs - epoch - 1
            total_batches_left = epochs_left * total_batches + batches_left
            total_time_left = total_batches_left * avg_batch_time

            # 打印批次进度和预计完成时间
            print(
                f"Epoch [{epoch + 1}/{args.epochs}], Batch [{batch_idx + 1}/{total_batches}], "
                f"Loss: {loss.item():.4f}, "
                f"预计本轮结束时间: {time.strftime('%H:%M:%S', time.gmtime(time_left))}, "
                # f"预计总训练结束时间: {time.strftime('%H:%M:%S', time.gmtime(total_time_left))}"
            )
            # 更新学习率
            if scheduler is not None:
                scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / total_batches
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{args.epochs}] 完成, 平均损失: {avg_loss:.4f}, "
              f"用时: {time.strftime('%H:%M:%S', time.gmtime(epoch_time))}")

        # 保存模型
        # 在保存模型部分添加处理clip_type中"/"的逻辑
        clip_type_name = args.clip_type.replace('/', '_')  # 将 'ViT-B/32' 转换为 'ViT-B_32'
        checkpoint_path = os.path.join(args.output_path,
                                       f"policy_model_{args.Modal}_{clip_type_name}_epoch_{epoch + 1}.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"模型已保存至 {checkpoint_path}")
    # 在训练循环结束后添加
    total_training_time = time.time() - start_time
    print(f"总训练时间: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
    print("训练完成")