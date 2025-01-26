import json
import argparse
from RL_base.cococaption.pycocotools.coco import COCO
from RL_base.cococaption.pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
from collections import defaultdict
import os
import sys
import random

def extract_short_answer(text, prompt_num=1):
    """
    从生成的文本中提取第二个Output:与英文句号之间的描述
    """
    if not text:
        return "no description"

    try:
        outputs = text.split("Output:")
        if len(outputs) >= prompt_num+2:
            generated_output = outputs[prompt_num+1]
            description = generated_output.split('.')[0].strip()
            return description if description else "no description"
    except Exception as e:
        print(f"提取描述时出错: {str(e)}")

    return "no description"

def parse_args():
    parser = argparse.ArgumentParser(description='计算生成结果的CIDEr分数')
    parser.add_argument('--results_file',
                        default='../log/ofv2_base/END_OUTPUT_linear/coco_caption/generation_results_temp_5000.json',
                        help='生成结果文件路径')
    parser.add_argument('--annotations_path',
                        default='/data16tb/ljq/datasets/coco_caption/annotations_captions_train2014.json',
                        help='COCO标注文件路径')
    parser.add_argument('--output_file',
                        default='/data16tb/ljq/Code/ICL_diversity_ofv3/data/coco_caption_labeled_cider.json',
                        help='输出文件路径')
    return parser.parse_args()


def load_coco_annotations(annotations_file):
    """加载COCO数据集的所有标注"""
    print("加载COCO标注...")
    coco = COCO(annotations_file)

    gts = defaultdict(list)
    for ann in coco.anns.values():
        gts[ann['image_id']].append(ann['caption'])

    return gts, coco


def process_results(results, all_gts, coco):
    """处理结果并计算CIDEr分数"""
    print("正在处理结果并计算CIDEr分数...")

    # 创建查询-示例字典
    query_examples = defaultdict(list)
    scorer = Cider()

    # 准备批量计算CIDEr的数据
    gts = {}
    res = {}
    result_map = {}  # 用于映射结果索引到原始数据

    for idx, item in enumerate(results):
        query_id = item['query_image_id']
        generated_caption = extract_short_answer(item['generated_text'])

        # 收集ground truth和生成的描述
        gts[idx] = all_gts[query_id]
        res[idx] = [generated_caption]
        result_map[idx] = (query_id, item)  # 保存映射关系

    # 批量计算CIDEr分数
    _, scores = scorer.compute_score(gts, res)

    # 将分数分配给对应的样本
    for idx, score in enumerate(scores):
        query_id, item = result_map[idx]
        generated_caption = extract_short_answer(item['generated_text'])

        query_examples[query_id].append({
            'example_image_id': item['example_image_id'],
            'example_image': item['example_filename'],
            'example_caption': item['demo_caption'],
            'generated_caption': generated_caption,
            'label': float(score),  # 使用批量计算的CIDEr分数
            'similarity_score': item['similarity_score']
        })

    # 构建新的数据集
    new_dataset = []
    for query_id, examples in query_examples.items():
        if examples:
            # 获取图像文件名
            img_info = coco.loadImgs([query_id])[0]

            new_dataset.append({
                'query_image_id': query_id,
                'query_image': f'COCO_train2014_{query_id:012d}.jpg',  # 使用image_id构建标准COCO文件名格式
                'query_captions': all_gts[query_id],  # 所有ground truth描述
                'examples': examples
            })

    return new_dataset


def print_stats(dataset):
    """打印数据集统计信息"""
    query_count = len(dataset)
    total_examples = sum(len(item['examples']) for item in dataset)

    print("\n数据集统计:")
    print(f"查询样本总数: {query_count}")
    print(f"示例样本总数: {total_examples}")
    print(f"平均每个查询的示例数: {total_examples / query_count:.2f}")

    # 计算平均CIDEr分数
    all_scores = [example['label']
                  for item in dataset
                  for example in item['examples']]
    avg_score = sum(all_scores) / len(all_scores)
    print(f"平均CIDEr分数: {avg_score:.4f}")
    print(f"最高CIDEr分数: {max(all_scores):.4f}")
    print(f"最低CIDEr分数: {min(all_scores):.4f}")


def main():
    args = parse_args()

    # 加载生成结果
    print(f"正在加载生成结果: {args.results_file}")
    with open(args.results_file, 'r') as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get('results', [])

    # 加载COCO标注
    all_gts, coco = load_coco_annotations(args.annotations_path)

    # 处理结果并构建新的数据集格式
    new_dataset = process_results(results, all_gts, coco)

    # 随机打乱数据集
    random.shuffle(new_dataset)

    # 打印统计信息
    print_stats(new_dataset)

    # 保存处理后的数据集
    print(f"正在保存结果到: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)

    print(f"处理完成！数据已保存到 {args.output_file}")


if __name__ == "__main__":
    main()