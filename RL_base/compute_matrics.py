import json
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_base.cococaption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from RL_base.cococaption.pycocoevalcap.bleu.bleu import Bleu
from RL_base.cococaption.pycocoevalcap.meteor.meteor import Meteor
from RL_base.cococaption.pycocoevalcap.rouge.rouge import Rouge
from RL_base.cococaption.pycocoevalcap.cider.cider import Cider


def load_results(results_file):
    """加载生成的结果文件"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['results']


def prepare_data(results):
    """准备评估所需的数据格式"""
    gts = {}
    res = {}

    for i, item in enumerate(results):
        # 修改ground truth格式
        gts[i] = [{'caption': cap} for cap in item['ground_truth_captions']]
        # 修改生成结果格式
        res[i] = [{'caption': item['generated_caption']}]

    return gts, res


def compute_metrics(gts, res):
    """计算各种评估指标"""
    # 初始化分词器
    tokenizer = PTBTokenizer()

    # 对参考描述和生成描述进行分词
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # 初始化评估器
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    # 存储评估结果
    eval_results = {}

    # 计算每个指标的分数
    for scorer, method in scorers:
        print(f"计算 {method} 分数...")
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_results[m] = sc
        else:
            eval_results[method] = score

    return eval_results


def main():
    # 加载结果文件
    # results_file = "/data16tb/ljq/Code/ICL_diversity_ofv3/log/ofv2_base/END_OUTPUT_linear/coco_caption/evaluation_results/final_results.json"  # 替换为你的结果文件路径
    # results_file = "/data16tb/ljq/Code/ICL_diversity_ofv3/log/ofv2_base/END_OUTPUT_linear/coco_caption/evaluation_results/temp_results_22286.json"  # 替换为你的结果文件路径
    results_file = "/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/caption_evaluation_results/multimodal_caption_eval_20250125_100243.json"  # 替换为你的结果文件路径

    results = load_results(results_file)

    # 准备数据
    gts, res = prepare_data(results)

    # 计算评估指标
    metrics = compute_metrics(gts, res)

    # 打印结果
    print("\n评估结果:")
    print(f"Bleu-1: {metrics['Bleu_1']:.4f}")
    print(f"Bleu-2: {metrics['Bleu_2']:.4f}")
    print(f"Bleu-3: {metrics['Bleu_3']:.4f}")
    print(f"Bleu-4: {metrics['Bleu_4']:.4f}")
    print(f"METEOR: {metrics['METEOR']:.4f}")
    print(f"ROUGE_L: {metrics['ROUGE_L']:.4f}")
    print(f"CIDEr: {metrics['CIDEr']:.4f}")

    # 保存评估结果
    # output_file = "/data16tb/ljq/Code/ICL_diversity_ofv3/log/ofv2_base/END_OUTPUT_linear/coco_caption/evaluation_results/final_results_score.json"
    output_file = "/data16tb/ljq/Code/ICL_diversity_ofv3/log/ofv2_base/END_OUTPUT_linear/coco_caption/evaluation_results/caption_generation_results_contrast_score.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n评估结果已保存至: {output_file}")


if __name__ == "__main__":
    main()