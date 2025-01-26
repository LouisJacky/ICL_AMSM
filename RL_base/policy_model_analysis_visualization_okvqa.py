import json
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 设置为非交互后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_result_path', type=str,
                        default="eval_results/all_methods_evaluation_20250119_212140.json")
    parser.add_argument('--train_image_dir', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/train2014")
    parser.add_argument('--val_image_dir', type=str,
                        default="/data16tb/ljq/datasets/ok_vqa/val2014")
    parser.add_argument('--output_dir', type=str, default="analysis_visualization")

    parser.add_argument('--max_cases', type=int, default=5)
    return parser.parse_args()


def load_and_process_results(eval_result_path):
    """加载评估结果并按方法分组"""
    with open(eval_result_path, 'r') as f:
        results = json.load(f)

    # 按question_id和方法组织结果
    organized_results = {}
    for result in results:
        qid = result['question_id']
        method = result['method']
        if qid not in organized_results:
            organized_results[qid] = {}
        organized_results[qid][method] = result

    return organized_results


def check_answer_correctness(result):
    """检查答案是否正确"""
    # 简化为直接字符串匹配
    return result['answer'].lower() in [ans.lower() for ans in result['gt_answers']]


def find_interesting_cases(organized_results, max_cases=5):
    """找出有趣的案例"""
    multimodal_better = []  # multimodal正确而其他错误的案例
    others_better = []  # 其他方法正确而multimodal错误的案例
    all_correct = []  # 所有方法都正确的案例

    for qid, methods in organized_results.items():
        if len(methods) != 3:  # 确保有所有三种方法的结果
            continue

        multimodal_correct = check_answer_correctness(methods['multimodal'])
        text_correct = check_answer_correctness(methods['text'])
        image_correct = check_answer_correctness(methods['image'])

        if multimodal_correct and (not text_correct or not image_correct):
            if len(multimodal_better) < max_cases:
                multimodal_better.append(methods)
                print(f"\nMultimodal Better Case {len(multimodal_better)}:")
                print(f"Question: {methods['multimodal']['question']}")
                print(f"GT Answers: {methods['multimodal']['gt_answers']}")
                print(f"Multimodal: {methods['multimodal']['answer']}")
                print(f"Text: {methods['text']['answer']}")
                print(f"Image: {methods['image']['answer']}")

        elif (text_correct or image_correct) and not multimodal_correct:
            if len(others_better) < max_cases:
                others_better.append(methods)
                print(f"\nOthers Better Case {len(others_better)}:")
                print(f"Question: {methods['multimodal']['question']}")
                print(f"GT Answers: {methods['multimodal']['gt_answers']}")
                print(f"Multimodal: {methods['multimodal']['answer']}")
                print(f"Text: {methods['text']['answer']}")
                print(f"Image: {methods['image']['answer']}")

        elif multimodal_correct and text_correct and image_correct:
            if len(all_correct) < max_cases:
                all_correct.append(methods)
                print(f"\nAll Correct Case {len(all_correct)}:")
                print(f"Question: {methods['multimodal']['question']}")
                print(f"GT Answers: {methods['multimodal']['gt_answers']}")
                print(f"Multimodal: {methods['multimodal']['answer']}")
                print(f"Text: {methods['text']['answer']}")
                print(f"Image: {methods['image']['answer']}")

    return multimodal_better, others_better, all_correct


def resize_image(image, target_size=(224, 224)):
    """调整图像大小为统一尺寸"""
    return image.resize(target_size, Image.Resampling.LANCZOS)

# def visualize_case(case, train_image_dir, val_image_dir, fig, gs, row_idx):
#     """可视化单个案例及其所有示例"""
#     # 加载并调整查询图像大小
#     query_image_path = os.path.join(val_image_dir,
#                                     f'COCO_val2014_{str(case["multimodal"]["image_id"]).zfill(12)}.jpg')
#     query_image = Image.open(query_image_path).convert('RGB')
#     query_image = resize_image(query_image)
#
#     # 创建查询图像的子图
#     ax_query = fig.add_subplot(gs[row_idx * 4:(row_idx + 1) * 4, 0])
#     ax_query.imshow(query_image)
#     ax_query.axis('off')
#
#     # 在查询图像上方添加问题和真实答案信息
#     question_text = (f"Q: {case['multimodal']['question']}\n"
#                      f"GT: {', '.join(case['multimodal']['gt_answers'][:3])}")
#     ax_query.set_title(question_text, fontsize=8, pad=5, wrap=True)
#
#     # 为每个方法显示所有示例图像
#     for col_idx, method in enumerate(['multimodal', 'text', 'image'], 1):
#         if method not in case:
#             continue
#
#         # 获取该方法的预测结果是否正确
#         is_correct = check_answer_correctness(case[method])
#         answer_color = 'green' if is_correct else 'red'
#
#         # 为每个示例创建子图
#         for example_idx, example in enumerate(case[method]['examples']):
#             # 创建示例图像的子图
#             ax = fig.add_subplot(gs[row_idx * 4 + example_idx, col_idx])
#
#             # 加载并调整示例图像大小
#             example_image_path = os.path.join(train_image_dir,
#                                               f'COCO_train2014_{str(example["image_id"]).zfill(12)}.jpg')
#             example_image = Image.open(example_image_path).convert('RGB')
#             example_image = resize_image(example_image)
#
#             ax.imshow(example_image)
#             ax.axis('off')
#
#             # 添加示例信息
#             if example_idx == 0:
#                 # 第一个示例上方显示方法名称和预测答案
#                 method_title = f"{method}\nPred: {case[method]['answer']}"
#                 ax.set_title(method_title, fontsize=8, color=answer_color, pad=5)
#
#             # 在每个示例右侧添加问题和答案信息
#             example_text = (f"Q: {example['question']}\n"
#                             f"A: {example['answer']}\n"
#                             #f"Score: {example['score']:.3f}"
#                             )
#             ax.text(1.02, 0.5, example_text,
#                     fontsize=6,
#                     transform=ax.transAxes,
#                     va='center',
#                     ha='left',
#                     wrap=True,
#                     bbox=dict(facecolor='white',
#                               alpha=0.8,
#                               edgecolor='none',
#                               pad=2))
#
#
# def save_visualizations(cases, case_type, args):
#     """保存可视化结果"""
#     if not cases:
#         print(f"没有找到{case_type}的案例")
#         return
#
#     n_cases = len(cases)
#     # 调整图像大小和间距，增加右侧空间用于显示文本
#     fig = plt.figure(figsize=(20, 6 * n_cases))  # 增加宽度
#     gs = GridSpec(n_cases * 4, 4, figure=fig,
#                   hspace=0.4,  # 增加行间距
#                   wspace=0.6,  # 增加列间距以容纳右侧文本
#                   left=0.05,  # 左边距
#                   right=0.95,  # 右边距
#                   top=0.95,  # 上边距
#                   bottom=0.05)  # 下边距
#
#     for i, case in enumerate(cases):
#         visualize_case(case, args.train_image_dir, args.val_image_dir, fig, gs, i)
#
#     # 保存结果
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = os.path.join(args.output_dir, f"{case_type}_{timestamp}.png")
#     plt.savefig(output_path, bbox_inches='tight', dpi=300)
#     plt.close()
#
#     print(f"已保存{case_type}的可视化结果到: {output_path}")

def visualize_case(case, train_image_dir, val_image_dir, fig, gs, row_idx):
    """可视化单个案例及其所有示例"""
    # 加载并调整查询图像大小
    query_image_path = os.path.join(val_image_dir,
                                    f'COCO_val2014_{str(case["multimodal"]["image_id"]).zfill(12)}.jpg')
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = resize_image(query_image)

    # 创建查询图像的子图
    ax_query = fig.add_subplot(gs[row_idx * 2:(row_idx + 1) * 2, 0])
    ax_query.imshow(query_image)
    ax_query.axis('off')

    # 在查询图像上方添加问题和真实答案信息
    question_text = (f"Q: {case['multimodal']['question']}\n"
                     f"GT: {', '.join(case['multimodal']['gt_answers'][:3])}")
    ax_query.set_title(question_text, fontsize=8, pad=5, wrap=True)

    # 为每个方法显示所有示例图像
    for col_idx, method in enumerate(['multimodal', 'text', 'image'], 1):
        if method not in case:
            continue

        # 获取该方法的预测结果是否正确
        is_correct = check_answer_correctness(case[method])
        answer_color = 'green' if is_correct else 'red'

        # 创建2x2网格的子图
        method_ax = fig.add_subplot(gs[row_idx * 2:(row_idx + 1) * 2, col_idx])
        method_ax.axis('off')

        # 在方法区域上方显示方法名称和预测答案
        method_title = f"{method}\nPred: {case[method]['answer']}"
        method_ax.set_title(method_title, fontsize=8, color=answer_color, pad=5)

        # 创建2x2的网格来放置示例图像
        grid_size = 2
        for example_idx, example in enumerate(case[method]['examples'][:4]):  # 限制为4个示例
            # 计算在2x2网格中的位置
            grid_row = example_idx // grid_size
            grid_col = example_idx % grid_size

            # 计算子图的位置和大小
            x = grid_col * 0.5
            y = 1 - (grid_row + 1) * 0.5
            w = 0.45
            h = 0.45

            # # 创建子图
            # example_ax = fig.add_axes([
            #     method_ax.get_position().x0 + x * method_ax.get_position().width,
            #     method_ax.get_position().y0 + y * method_ax.get_position().height,
            #     w * method_ax.get_position().width,
            #     h * method_ax.get_position().height
            # ])
            #
            # # 加载并显示示例图像
            # example_image_path = os.path.join(train_image_dir,
            #                                   f'COCO_train2014_{str(example["image_id"]).zfill(12)}.jpg')
            # example_image = Image.open(example_image_path).convert('RGB')
            # example_image = resize_image(example_image)
            # example_ax.imshow(example_image)
            # example_ax.axis('off')
            #
            # # 在图像上方添加问答信息
            # example_text = (f"Q: {example['question']}\nA: {example['answer']}")
            # example_ax.set_title(example_text, fontsize=6, wrap=True, pad=2)

            # 创建子图
            example_ax = fig.add_axes([
                method_ax.get_position().x0 + x * method_ax.get_position().width,
                method_ax.get_position().y0 + y * method_ax.get_position().height,
                w * method_ax.get_position().width,
                h * method_ax.get_position().height
            ])

            # 加载并显示示例图像
            example_image_path = os.path.join(train_image_dir,
                                              f'COCO_train2014_{str(example["image_id"]).zfill(12)}.jpg')
            example_image = Image.open(example_image_path).convert('RGB')
            example_image = resize_image(example_image)
            example_ax.imshow(example_image)
            example_ax.axis('off')

            # 在图像上方添加问答信息，使用textwrap进行文本换行
            import textwrap
            max_width = 40  # 设置每行最大字符数
            wrapped_question = textwrap.fill(f"Q: {example['question']}", width=max_width)
            wrapped_answer = textwrap.fill(f"A: {example['answer']}", width=max_width)
            example_text = f"{wrapped_question}\n{wrapped_answer}"

            # 调整标题位置和样式
            example_ax.set_title(example_text,
                                 fontsize=6,
                                 pad=2,
                                 wrap=True,
                                 bbox=dict(
                                     facecolor='white',
                                     alpha=0.8,
                                     edgecolor='none',
                                     pad=2
                                 ))


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
    multimodal_better, others_better, all_correct = find_interesting_cases(organized_results, max_cases=args.max_cases)

    # 可视化并保存结果
    print("生成可视化结果...")
    save_visualizations(multimodal_better, "multimodal_better", args)
    save_visualizations(others_better, "others_better", args)
    save_visualizations(all_correct, "all_correct", args)

    print("分析完成！")


if __name__ == "__main__":
    main()