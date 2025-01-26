import json
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 设置为非交互后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_retrieval_results(results_file, train_image_dir, test_image_dir, num_queries=4, num_examples=4):
    """可视化检索结果的前几个查询样本及其匹配的示例"""

    # 读取结果文件
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 获取前num_queries个查询
    query_ids = list(results.keys())[:num_queries]

    # 创建大图
    fig = plt.figure(figsize=(20, 5 * num_queries))
    gs = gridspec.GridSpec(num_queries, num_examples + 1, width_ratios=[1] * 5)

    for query_idx, query_id in enumerate(query_ids):
        query_data = results[query_id]

        # 加载查询图片
        query_image_id = query_data['query_image_id']
        query_image_path = os.path.join(test_image_dir, f"COCO_val2014_{query_image_id:012d}.jpg")
        query_image = Image.open(query_image_path)

        # 显示查询图片
        ax = plt.subplot(gs[query_idx, 0])
        ax.imshow(query_image)
        ax.axis('off')
        ax.set_title(f"Query:\n{query_data['query_question']}\nAnswer: {query_data['query_most_common_answer']}",
                     wrap=True)

        # 显示top-k示例
        for k in range(num_examples):
            example = query_data['top_examples'][k]
            example_image_id = example['example_image_id']
            example_image_path = os.path.join(train_image_dir, f"COCO_train2014_{example_image_id:012d}.jpg")
            example_image = Image.open(example_image_path)

            ax = plt.subplot(gs[query_idx, k + 1])
            ax.imshow(example_image)
            ax.axis('off')
            ax.set_title(f"Example {k + 1} (score: {example['score']:.3f}):\n{example['example_question']}\n"
                         f"Answer: {example['example_most_common_answer']}",
                         wrap=True)

    plt.tight_layout()

    # 保存结果
    output_dir = os.path.dirname(results_file)
    output_path = os.path.join(output_dir, 'visualization.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    # 设置路径
    results_file = "../log/ofv2_base/END_OUTPUT_linear/okvqa/evaluation_results/retrieval_results_okvqa_20250115_155158.json"  # 替换为你的结果文件路径
    train_image_dir = "/data16tb/ljq/datasets/ok_vqa/train2014"
    test_image_dir = "/data16tb/ljq/datasets/ok_vqa/val2014"

    visualize_retrieval_results(results_file, train_image_dir, test_image_dir)