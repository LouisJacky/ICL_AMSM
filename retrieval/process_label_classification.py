import json
import random
from tqdm import tqdm
from collections import defaultdict
import os
from retrieval.dataset import TinyImageNetDataset  # 添加这行导入

def print_stats(new_dataset):
    query_count = len(new_dataset)
    total_examples = sum(len(item['examples']) for item in new_dataset)

    print("\n数据集统计:")
    print(f"查询样本总数: {query_count}")
    print(f"示例样本总数: {total_examples}")
    print(f"平均每个查询的示例数: {total_examples / query_count:.2f}")


def get_image_name(idx, dataset):
    """根据索引获取图像名称

    Args:
        idx: 数据集中的索引
        dataset_root: 数据集根目录

    Returns:
        str: 图像相对路径
    """
    # 使用 TinyImageNetDataset 来获取图像路径

    sample = dataset[idx]
    # 从完整路径中提取相对路径
    full_path = sample['img_path']
    relative_path = '/'.join(full_path.split('/')[-3:])  # 获取 'class/images/image.JPEG' 格式的路径
    return relative_path


def main():
    # 配置参数
    dataset_root = '/path/to/datasets/tiny-imagenet-200'
    input_file = '../log/ofv2_base/END_OUTPUT_linear/tiny_imagenet/tiny_imagenet_label_confidence_4800_32.json'
    output_file = '../data/tiny_imagenet_labeled_confidence.json'

    # 读取标签对文件
    with open(input_file, 'r') as f:
        pair_data = json.load(f)

    # 创建查询示例字典
    query_examples = defaultdict(list)
    dataset = TinyImageNetDataset(dataset_root, split='train')

    # 处理数据
    for pair in tqdm(pair_data['samples'], desc="收集示例"):
        query_idx = pair['query_idx']
        example_idx = pair['example_idx']

        # 获取图像名称
        query_image = get_image_name(query_idx, dataset)
        example_image = get_image_name(example_idx, dataset)

        if query_image and example_image:
            query_examples[query_image].append({
                'example_image': example_image,
                'example_label': pair['example_label'],
                'query_label': pair['query_label'],
                'confidence_score': pair['confidence_score'],
                'similarity_score': pair['similarity_score']
            })

    # 构建新的数据集
    new_dataset = []
    for query_image, examples in tqdm(query_examples.items(), desc="构建数据集"):
        if examples:
            # 从第一个示例中获取查询标签（所有示例的query_label都相同）
            query_label = examples[0]['query_label']
            new_dataset.append({
                'query_image': query_image,
                'query_label': query_label,
                'examples': examples
            })

    # 随机打乱数据集
    random.shuffle(new_dataset)

    # 打印统计信息
    print_stats(new_dataset)

    # 保存处理后的数据集
    with open(output_file, 'w') as f:
        json.dump(new_dataset, f, indent=2)

    print(f"已将数据集保存到 {output_file}")


if __name__ == "__main__":
    main()