import numpy as np
import h5py
import json
import os
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str,
                        default='/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/coco_features.h5',
                        help='COCO特征文件路径')
    parser.add_argument('--clustering_dir', type=str,
                        default='/data16tb/ljq/Code/ICL_diversity_ofv3/retrieval/clustering_results_coco',
                        help='聚类结果目录')
    parser.add_argument('--output_file', type=str,
                        default='selected_coco_samples.json',
                        help='输出文件路径')
    parser.add_argument('--n_clusters', type=int,
                        default=5000,
                        )
    return parser.parse_args()


def load_data(args):
    """加载特征、图像ID和聚类结果"""
    print("正在加载数据...")

    # 加载特征和图像ID
    with h5py.File(args.features_file, 'r') as f:
        features = np.array(f['image_features'][:])
        image_ids = f['image_ids'][:]

    # 加载聚类结果
    cluster_dir = os.path.join(args.clustering_dir, f'clusters_{args.n_clusters}')
    cluster_labels = np.load(os.path.join(cluster_dir, 'cluster_labels.npy'))
    centroids = np.load(os.path.join(cluster_dir, 'centroids.npy'))

    return features, image_ids, cluster_labels, centroids


def select_closest_samples(features, image_ids, cluster_labels, centroids):
    """为每个簇选择距离中心最近的样本"""
    print("正在选择最近样本...")

    selected_samples = {}
    unique_clusters = range(len(centroids))  # 簇的编号从0到n_clusters-1

    for cluster_id in tqdm(unique_clusters, desc="处理聚类"):
        # 获取当前簇的样本
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features[cluster_mask]
        cluster_image_ids = image_ids[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_features) == 0:
            continue

        # 计算到中心的距离
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)

        # 选择最近的样本
        closest_idx = np.argmin(distances)

        # 保存结果：图像ID和特征索引
        selected_samples[int(cluster_id)] = {
            'image_id': int(cluster_image_ids[closest_idx]),
            # 'feature_idx': int(cluster_indices[closest_idx]),
            'distance': float(distances[closest_idx])
        }

    return selected_samples


def save_results(selected_samples, args):
    """保存选定的样本信息"""
    print("正在保存结果...")

    # 添加元数据
    output_data = {
        'n_clusters': args.n_clusters,
        'total_samples': len(selected_samples),
        'samples': selected_samples
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"结果已保存至: {args.output_file}")


def main():
    args = parse_args()

    # 加载数据
    features, image_ids, cluster_labels, centroids = load_data(args)

    # 选择样本
    selected_samples = select_closest_samples(
        features, image_ids, cluster_labels, centroids
    )

    # 保存结果
    save_results(selected_samples, args)

    # 打印统计信息
    print(f"\n总共选择了 {len(selected_samples)} 个样本")

    # 计算平均距离
    avg_distance = np.mean([s['distance'] for s in selected_samples.values()])
    print(f"所选样本到中心点的平均距离: {avg_distance:.4f}")


if __name__ == "__main__":
    main()