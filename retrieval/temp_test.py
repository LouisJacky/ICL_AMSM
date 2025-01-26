import h5py
import torch
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.dataset import TinyImageNetDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/data16tb/ljq/datasets/tiny-imagenet-200')
    parser.add_argument('--features_file', type=str, default='/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/tiny_imagenet_features.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=100, help='验证的样本数量，-1表示全部验证')
    parser.add_argument('--gpu', type=str, default='6')
    return parser.parse_args()


def verify_features():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载CLIP模型
    print("加载CLIP模型...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 加载数据集
    print("加载数据集...")
    dataset = TinyImageNetDataset(args.root_dir, split='train')

    # 加载特征文件
    print("加载特征文件...")
    with h5py.File(args.features_file, 'r') as f:
        stored_features = torch.from_numpy(f['features'][:])

    # 确定验证的样本数量
    num_samples = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))

    print(f"\n开始验证前{num_samples}个样本的特征...")
    mismatches = []

    with torch.no_grad():
        for idx in tqdm(range(num_samples)):
            # 获取数据集中的图像
            sample = dataset[idx]
            image = preprocess(sample['image']).unsqueeze(0).to(device)

            # 使用CLIP提取当前图像的特征
            current_feature = clip_model.encode_image(image).cpu().float()

            # 获取存储的特征
            stored_feature = stored_features[idx:idx + 1]

            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(current_feature, stored_feature)

            # 如果相似度不够高，记录不匹配
            if similarity < 0.9999:  # 允许有极小的数值误差
                mismatches.append({
                    'index': idx,
                    'image_path': sample['image'].filename,
                    'similarity': similarity.item()
                })

            # 每验证100个样本输出一次进度
            if (idx + 1) % 10 == 0:
                print(f"\n已验证 {idx + 1} 个样本")
                if mismatches:
                    print(f"当前发现 {len(mismatches)} 个不匹配")

    # 输出验证结果
    print("\n验证完成！")
    print(f"总共验证了 {num_samples} 个样本")

    if not mismatches:
        print("所有特征完全匹配！✅")
    else:
        print(f"发现 {len(mismatches)} 个不匹配的样本：")
        for mismatch in mismatches:
            print(f"\n索引: {mismatch['index']}")
            print(f"图像路径: {mismatch['image_path']}")
            print(f"特征相似度: {mismatch['similarity']:.6f}")


if __name__ == "__main__":
    verify_features()