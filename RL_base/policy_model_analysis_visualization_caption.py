import json
import os
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_result_path', type=str,
                        default="caption_evaluation_results/all_methods_caption_eval_20250119_221326.json")
    parser.add_argument('--train_image_dir', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/train2014")
    parser.add_argument('--val_image_dir', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/val2014")
    parser.add_argument('--output_dir', type=str, default="caption_analysis_visualization")
    parser.add_argument('--max_cases', type=int, default=5)
    return parser.parse_args()


def load_and_process_results(eval_result_path):
    """加载评估结果并按方法分组"""
    with open(eval_result_path, 'r') as f:
        results = json.load(f)

    # 按image_id和方法组织结果
    organized_results = {}
    for method_results in results:
        method = method_results['method']
        for result in method_results['results']:
            img_id = result['query_image_id']
            if img_id not in organized_results:
                organized_results[img_id] = {}
            organized_results[img_id][method] = result

    return organized_results


def find_interesting_cases(organized_results, max_cases=5):
    """找出有趣的案例"""
    multimodal_better = []  # multimodal得分更高的案例
    image_better = []  # image方法得分更高的案例
    similar_performance = []  # 两种方法表现相近的案例

    for img_id, methods in organized_results.items():
        if len(methods) != 2:  # 确保有两种方法的结果
            continue

        multimodal_score = methods['multimodal']['cider_score']
        image_score = methods['image']['cider_score']
        score_diff = multimodal_score - image_score

        # 根据分数差异分类案例
        if score_diff > 0.5 and len(multimodal_better) < max_cases:
            multimodal_better.append(methods)
            print(f"\nMultimodal Better Case {len(multimodal_better)}:")
            print(f"Generated (Multimodal): {methods['multimodal']['generated_caption']}")
            print(f"Generated (Image): {methods['image']['generated_caption']}")
            print(f"Ground Truth: {methods['multimodal']['ground_truth_captions'][0]}")
            print(f"Scores - Multimodal: {multimodal_score:.3f}, Image: {image_score:.3f}")

        elif score_diff < -0.5 and len(image_better) < max_cases:
            image_better.append(methods)
            print(f"\nImage Better Case {len(image_better)}:")
            print(f"Generated (Multimodal): {methods['multimodal']['generated_caption']}")
            print(f"Generated (Image): {methods['image']['generated_caption']}")
            print(f"Ground Truth: {methods['multimodal']['ground_truth_captions'][0]}")
            print(f"Scores - Multimodal: {multimodal_score:.3f}, Image: {image_score:.3f}")

        elif abs(score_diff) <= 0.2 and len(similar_performance) < max_cases:
            similar_performance.append(methods)
            print(f"\nSimilar Performance Case {len(similar_performance)}:")
            print(f"Generated (Multimodal): {methods['multimodal']['generated_caption']}")
            print(f"Generated (Image): {methods['image']['generated_caption']}")
            print(f"Ground Truth: {methods['multimodal']['ground_truth_captions'][0]}")
            print(f"Scores - Multimodal: {multimodal_score:.3f}, Image: {image_score:.3f}")

    return multimodal_better, image_better, similar_performance


def resize_image(image, target_size=(224, 224)):
    """调整图像大小为统一尺寸"""
    return image.resize(target_size, Image.Resampling.LANCZOS)


def visualize_case(case, train_image_dir, val_image_dir, fig, gs, row_idx):
    """可视化单个案例及其所有示例"""
    # 加载并调整查询图像大小
    query_image_path = os.path.join(val_image_dir, case['multimodal']['query_file_name'])
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = resize_image(query_image)

    # 创建查询图像的子图
    ax_query = fig.add_subplot(gs[row_idx * 2:(row_idx + 1) * 2, 0])
    ax_query.imshow(query_image)
    ax_query.axis('off')

    # 在查询图像上方添加真实caption
    import textwrap
    max_width = 50  # 设置每行最大字符数
    gt_caption = textwrap.fill(f"GT: {case['multimodal']['ground_truth_captions'][0]}",
                              width=max_width)
    ax_query.set_title(gt_caption,
                      fontsize=8,
                      pad=5,
                      bbox=dict(facecolor='white',
                              alpha=0.9,
                              edgecolor='none',
                              pad=3))

    # 为每个方法显示示例图像和生成的caption
    for col_idx, method in enumerate(['multimodal', 'image'], 1):
        if method not in case:
            continue

        # 创建方法区域
        method_ax = fig.add_subplot(gs[row_idx * 2:(row_idx + 1) * 2, col_idx])
        method_ax.axis('off')

        # 添加方法标题和生成的caption
        # method_caption = textwrap.fill(
        #     f"{method.capitalize()} Generated Caption (CIDEr: {case[method]['cider_score']:.3f}):\n"
        #     f"{case[method]['generated_caption']}",
        #     width=max_width)
        # 添加方法标题和生成的caption
        title_part = f"{method.capitalize()} Generated Caption"
        caption_part = textwrap.fill(case[method]['generated_caption'], width=max_width)
        score_part = f"(CIDEr: {case[method]['cider_score']*100:.3f}):"
        method_caption = f"{title_part}\nPred:{caption_part}\n{score_part}"
        method_ax.set_title(method_caption,
                          fontsize=8,
                          pad=10,
                          bbox=dict(facecolor='white',
                                  alpha=0.9,
                                  edgecolor='none',
                                  pad=2))

        # 创建2x2网格显示示例图像
        grid_size = 2
        for example_idx, example in enumerate(case[method]['examples'][:4]):
            # 计算在2x2网格中的位置
            grid_row = example_idx // grid_size
            grid_col = example_idx % grid_size

            # 计算子图的位置和大小
            x = grid_col * 0.5
            y = 1 - (grid_row + 1) * 0.5
            w = 0.45
            h = 0.45

            # 创建子图
            example_ax = fig.add_axes([
                method_ax.get_position().x0 + x * method_ax.get_position().width,
                method_ax.get_position().y0 + y * method_ax.get_position().height,
                w * method_ax.get_position().width,
                h * method_ax.get_position().height
            ])

            # 加载并显示示例图像
            example_image_path = os.path.join(train_image_dir, example['example_file_name'])
            example_image = Image.open(example_image_path).convert('RGB')
            example_image = resize_image(example_image)
            example_ax.imshow(example_image)
            example_ax.axis('off')

            # 在图像上方添加示例caption
            example_caption = example['example_captions'][0]
            wrapped_caption = textwrap.fill(
                example_caption[:100] + ('...' if len(example_caption) > 100 else ''),
                width=40)
            example_ax.set_title(wrapped_caption,
                               fontsize=6,
                               pad=2,
                               wrap=True,
                               bbox=dict(facecolor='white',
                                       alpha=0.9,
                                       edgecolor='none',
                                       pad=2))


def save_visualizations(cases, case_type, args):
    """保存可视化结果"""
    if not cases:
        print(f"没有找到{case_type}的案例")
        return

    n_cases = len(cases)
    # 调整图像大小和布局
    fig = plt.figure(figsize=(20, 8 * n_cases))
    gs = GridSpec(n_cases * 2, 4, figure=fig,
                  hspace=0.3,
                  wspace=0.3,
                  left=0.05,
                  right=0.95,
                  top=0.95,
                  bottom=0.05)

    for i, case in enumerate(cases):
        visualize_case(case, args.train_image_dir, args.val_image_dir, fig, gs, i)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{case_type}_{timestamp}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"已保存{case_type}的可视化结果到: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载和处理结果
    print("加载评估结果...")
    organized_results = load_and_process_results(args.eval_result_path)

    # 找出有趣的案例
    print("分析案例...")
    multimodal_better, image_better, similar_performance = find_interesting_cases(
        organized_results, max_cases=args.max_cases)

    # 可视化并保存结果
    print("生成可视化结果...")
    save_visualizations(multimodal_better, "multimodal_better", args)
    save_visualizations(image_better, "image_better", args)
    save_visualizations(similar_performance, "similar_performance", args)

    print("分析完成！")


if __name__ == "__main__":
    main()