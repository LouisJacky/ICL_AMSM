import h5py
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str,
                        default='/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/coco_features.h5',
                        help='COCO特征文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='clustering_results_coco',
                        help='结果输出目录')
    parser.add_argument('--n_clusters_list', nargs='+', type=int,
                        default=[7000, 8000, 9000, 10000, 11000],
                        help='要尝试的聚类数列表')
    return parser.parse_args()


def load_features(features_file):
    """加载COCO特征"""
    print("正在加载特征...")
    with h5py.File(features_file, 'r') as f:
        features = torch.from_numpy(f['image_features'][:])
        image_ids = f['image_ids'][:]
    return features, image_ids


def perform_clustering_experiment(features, n_clusters_list):
    """对不同的聚类数进行实验"""
    results = {}
    avg_distances = []

    for n_clusters in tqdm(n_clusters_list, desc="进行不同聚类数的实验"):
        # 执行聚类
        # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
        cluster_labels = kmeans.fit_predict(features)

        # # 计算样本到簇中心的平均距离
        # distances = np.min(
        #     np.linalg.norm(
        #         features[:, np.newaxis] - kmeans.cluster_centers_,
        #         axis=2
        #     ),
        #     axis=1
        # )
        # avg_distance = float(np.mean(distances))
        # 计算样本到簇中心的平均距离
        all_distances = []
        batch_size = 1000  # 每批处理的样本数
        centroids = kmeans.cluster_centers_

        for i in tqdm(range(0, len(features), batch_size), desc="计算到中心点距离"):
            # 获取当前批次的特征
            batch_features = features[i:i + batch_size]
            # 计算当前批次中每个样本到所有中心点的距离
            batch_distances = np.min(
                np.linalg.norm(
                    batch_features[:, np.newaxis] - centroids,
                    axis=2
                ),
                axis=1
            )
            all_distances.extend(batch_distances)

        avg_distance = float(np.mean(all_distances))

        # 保存结果
        results[n_clusters] = {
            'cluster_labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'avg_distance': avg_distance
        }
        avg_distances.append(avg_distance)

        print(f"聚类数 {n_clusters}: 平均距离 = {avg_distance:.4f}")

    return results, avg_distances


def save_results(results, summary, output_dir):
    """保存聚类结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存汇总信息
    with open(os.path.join(output_dir, 'clustering_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # 保存每个聚类结果
    for n_clusters, result in results.items():
        cluster_dir = os.path.join(output_dir, f'clusters_{n_clusters}')
        os.makedirs(cluster_dir, exist_ok=True)

        # 保存聚类标签
        np.save(
            os.path.join(cluster_dir, 'cluster_labels.npy'),
            result['cluster_labels']
        )
        # 保存聚类中心
        np.save(
            os.path.join(cluster_dir, 'centroids.npy'),
            result['centroids']
        )


def main():
    args = parse_args()

    # 加载特征
    features, image_ids = load_features(args.features_file)

    # 进行聚类实验
    results, avg_distances = perform_clustering_experiment(
        features,
        args.n_clusters_list
    )

    # 保存结果
    summary = {
        'n_clusters_list': args.n_clusters_list,
        'avg_distances': avg_distances
    }
    save_results(results, summary, args.output_dir)

    print(f"\n聚类实验完成！结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()