import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import os
import random
import numpy as np
import UTILS as utils
import clip
from policy_model import ResCaptionMatchingModel
import time


class CaptionDataset(Dataset):
    def __init__(self, json_file, image_dir, num_pairs=20):
        with open(json_file, 'r') as f:
            raw_data = json.load(f)

        # 重新组织数据结构
        self.data = []
        for item in raw_data:
            query_info = {
                'query_image': item['query_image'],
                'query_captions': item['query_captions'],
                'examples': item['examples']
            }
            self.data.append(query_info)

        self.image_dir = image_dir
        self.num_pairs = num_pairs

        # 为每个query创建示例池
        self.example_pools = []
        for item in self.data:
            total_examples = len(item['examples'])
            self.example_pools.append({
                'total': total_examples,
                'used': set(),
                'current': []
            })
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        for pool in self.example_pools:
            total_examples = pool['total']
            used_examples = pool['used']

            if total_examples - len(used_examples) < self.num_pairs:
                used_examples.clear()

            available = list(set(range(total_examples)) - used_examples)
            selected = random.sample(available, min(self.num_pairs, len(available)))

            used_examples.update(selected)
            pool['current'] = selected

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        all_examples = item['examples']
        selected_indices = self.example_pools[idx]['current']
        selected_examples = [all_examples[i] for i in selected_indices]

        query_images = []
        example_images = []
        labels = []

        # 加载query图像(只需加载一次)
        query_image = Image.open(os.path.join(self.image_dir, item['query_image']))

        for example in selected_examples:
            example_image = Image.open(os.path.join(self.image_dir, example['example_image']))

            query_images.append(query_image)
            example_images.append(example_image)
            labels.append(example['label'])

        return {
            'query_image': query_images,
            'example_image': example_images,
            'label': torch.tensor(labels, dtype=torch.float32),
        }


def custom_collate(batch):
    query_images = [img for item in batch for img in item['query_image']]
    example_images = [img for item in batch for img in item['example_image']]
    labels = torch.cat([item['label'] for item in batch])

    return {
        'query_image': query_images,
        'example_image': example_images,
        'label': labels,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_feature_dim', type=int, default=512, help='Policy network final layer hidden state size.')
    parser.add_argument('--policy_num_heads', type=int, default=8,
                        help='Policy network')
    parser.add_argument('--policy_num_layers', type=int, default=2,
                        help='Policy network')
    parser.add_argument('--policy_dropout', type=float, default=0.0,
                        help='Policy network')

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
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--base_dir', default='../datasets',
                        help='pascal base dir')


    parser.add_argument('--task', default='caption', choices=['vqa', 'caption'])
    parser.add_argument('--BENCHMARK', default='coco_caption', type=str,
                        help='dataset type:"vizwiz, pair, pascal, fss, coco_caption, iSALD, ottawa, aeroscapes"')

    # 尝试调整学习率
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs.')

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
                        default="/path/to/AMSM/log/ofv2_base/END_OUTPUT_linear/vizwiz/policy_model_epoch_2.pth",
                        help='Learned model checkpoint path')

    parser.add_argument('--backbone', default="clip", type=str, help='backbone of retrival feature VIT、clip')

    parser.add_argument('--clip_ckpt_path', type=str,
                        default='../pretrained_model/ViT-B-32.pt',
                        help='CLIP Pre-training model checkpoint path.')

    parser.add_argument('--train_VIT_Img_size', default=224, type=int, help='input size of VIT')

    parser.add_argument('--mode', default='train', type=str, help='train, val')
    parser.add_argument('--provide_ex_ans', type=bool, default=False, help='provide example answer')

    # ofv2
    ## VizWiz_labeled Dataset
    parser.add_argument(
        "--json_file",
        type=str,
        help="Path to the vizwiz questions json file.",
        default="/path/to/AMSM/data/coco_caption_labeled_cider.json",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the COCO train2014 images directory",
        default="/path/to/datasets/ok_vqa/train2014",
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    utils.create_dir(args.output_path)
    return args

def preprocess_batch(images, preprocess_fn):
    return torch.stack([preprocess_fn(image) for image in images])

# 添加自定义损失函数
def custom_loss(outputs, labels, num_pairs):
    num_query = len(outputs) // num_pairs
    total_loss = 0.0

    for i in range(num_query):
        start_idx = i * num_pairs
        end_idx = (i + 1) * num_pairs
        query_outputs = outputs[start_idx:end_idx]
        query_labels = labels[start_idx:end_idx]

        mean_label = torch.mean(query_labels)
        above_mean_mask = query_labels >= mean_label
        below_mean_mask = query_labels <= mean_label

        above_mean_mean = (query_outputs[above_mean_mask]).mean()
        below_mean_mean = (query_outputs[below_mean_mask]).mean()

        difference = above_mean_mean - below_mean_mean
        print(f"高均值: {above_mean_mean:.4f}, "
              f"低均值: {below_mean_mean:.4f}, "
              f"差值: {difference:.4f}")

        pair_losses = []
        for j in range(num_pairs):
            for k in range(j + 1, num_pairs):
                pair_outputs = torch.stack([query_outputs[j], query_outputs[k]])
                pair_labels = torch.stack([query_labels[j], query_labels[k]])
                pair_mean_label = torch.mean(pair_labels)
                pair_above_mean_mask = pair_labels >= pair_mean_label
                pair_output_probs = torch.nn.functional.softmax(pair_outputs, dim=-1)
                pair_loss = (pair_mean_label - pair_labels) * (torch.log(pair_output_probs))
                pair_losses.append(torch.mean(pair_loss[pair_above_mean_mask]))

        # total_loss += torch.sum(torch.stack(pair_losses))
        total_loss += torch.mean(torch.stack(pair_losses))

    return total_loss / num_query

if __name__ == '__main__':
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    dataset = CaptionDataset(args.json_file, args.image_dir, num_pairs=args.candidate_num)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=custom_collate)

    # 初始化模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    policy_model = ResCaptionMatchingModel(clip_model, hidden_size=args.policy_feature_dim).to(device)

    # 优化器
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr, weight_decay=1e-5)
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
            query_images = preprocess_batch(batch['query_image'], preprocess).to(device)
            example_images = preprocess_batch(batch['example_image'], preprocess).to(device)

            labels = batch['label'].to(device)

            # 前向传播
            outputs = policy_model(query_images, example_images)

            # 使用自定义损失函数计算损失
            loss = custom_loss(outputs, labels, args.candidate_num)

            # 在训练循环中添加调试信息
            # print(f"输出形状: {outputs.shape}, 标签形状: {labels.shape}")
            print(f"输出样本: {outputs[:min(5,len(outputs))]}")
            print(f"标签样本: {labels[:min(5,len(labels))]}")

            # # 蒙特卡洛采样
            # # 获取当前batch的实际大小
            # current_batch_size = len(query_images) // args.candidate_num
            # outputs = outputs.reshape(current_batch_size, args.candidate_num)
            #
            #
            # # 应用softmax获取概率分布
            # probs = torch.nn.functional.softmax(outputs, dim=1)
            # selected_indices = []
            #
            # for i in range(current_batch_size):
            #     # 将概率转换为numpy数组
            #     cand_prob = probs[i].detach().cpu().numpy()
            #     # 使用np.random.choice进行无放回采样
            #     sample_indices = np.random.choice(
            #         range(args.candidate_num),
            #         args.select_number,
            #         p=cand_prob,
            #         replace=False
            #     )
            #     selected_indices.extend(sample_indices + i * args.candidate_num)
            #
            # # 获取选中的outputs和labels
            # selected_outputs = outputs.reshape(-1)[selected_indices]
            # selected_labels = labels[selected_indices]
            # loss = custom_loss(selected_outputs, selected_labels, args.select_number)
            #
            # print(f"输出样本: {selected_outputs[:min(5, len(selected_outputs))]}")
            # print(f"标签样本: {selected_labels[:min(5, len(selected_labels))]}")

            # if loss.item() != 0:  # 只在loss不为0时进行反向传播

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
        checkpoint_path = os.path.join(args.output_path, f"policy_model_epoch_{epoch + 1}.pth")
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