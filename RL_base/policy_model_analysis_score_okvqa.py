import os
import json
import argparse
import torch
from PIL import Image
import sys
from tqdm import tqdm
import math
from datetime import datetime
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_flamingo import create_model_and_transforms
from open_flamingo.inference import ofv2_inference_n_promt
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy

import re

def parse_args():
    parser = argparse.ArgumentParser()
    # OFv2模型参数
    parser.add_argument('--lm_path', type=str, default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str, default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str, default="/path/to/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--shot_number', type=int, default=4)

    # 数据路径
    parser.add_argument('--val_image_dir', type=str, default="/path/to/datasets/ok_vqa/val2014")
    parser.add_argument('--train_image_dir', type=str, default="/path/to/datasets/ok_vqa/train2014")
    parser.add_argument('--question_json_path', type=str,
                        default="/path/to/datasets/ok_vqa/OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument('--annotation_json_path', type=str,
                        default="/path/to/datasets/ok_vqa/mscoco_val2014_annotations.json")

    # 分析结果路径
    parser.add_argument('--analysis_dir', type=str, default="analysis_results")

    parser.add_argument('--okvqa_train_questions_json_path', type=str,
                        default="/path/to/datasets/ok_vqa/OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument('--okvqa_train_annotations_json_path', type=str,
                        default="/path/to/datasets/ok_vqa/mscoco_train2014_annotations.json")
    parser.add_argument('--output_dir', type=str,
                        default="eval_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return parser.parse_args()


def extract_short_answer(text):
    """从文本中提取简短答案"""
    pattern = r'Short answer:\s*(.*?)\.'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    else:
        pattern = r'Answer:\s*(.*?)\.'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def load_matching_results(analysis_dir, method, args):
    """加载指定方法的匹配结果"""
    matching_files = [f for f in os.listdir(analysis_dir) if f.startswith(f'{method}_matching_')]
    if not matching_files:
        raise FileNotFoundError(f"未找到{method}方法的匹配结果文件")

    # 使用最新的结果文件
    latest_file = max(matching_files, key=lambda x: os.path.getctime(os.path.join(analysis_dir, x)))

    # 加载测试集问题和答案数据
    with open(args.question_json_path, 'r') as f:
        questions_data = json.load(f)['questions']
        qid_to_question = {str(q['question_id']): q for q in questions_data}

    with open(args.annotation_json_path, 'r') as f:
        annotations_data = json.load(f)['annotations']
        qid_to_annotation = {str(ann['question_id']): ann for ann in annotations_data}

    # 加载匹配结果并添加查询相关信息
    with open(os.path.join(analysis_dir, latest_file), 'r') as f:
        matching_results = json.load(f)
        # 为每个查询样本添加完整信息
        for qid in matching_results['results']:
            question_data = qid_to_question[qid]
            annotation_data = qid_to_annotation[qid]
            matching_results['results'][qid].update({
                'query_image_id': question_data['image_id'],
                'query_question': question_data['question'],
                'query_answers': [ans['answer'] for ans in annotation_data['answers']]
            })

    return matching_results

def evaluate_method(method_results, args, ofv2_model, image_processor, tokenizer, device):
    """评估特定方法的性能"""
    results = []
    total_samples = len(method_results['results'])
    progress_step = math.ceil(total_samples * 0.1)  # 每10%计算一次准确率

    for i, (query_id, item) in enumerate(tqdm(method_results['results'].items(),
                                            desc=f"评估{method_results['metadata']['method']}方法")):
        # 准备演示样本
        demo_image_list = []
        demo_question_list = []
        demo_answer_list = []

        top_examples = item['top_examples'][:args.shot_number]
        top_examples.reverse()
        for example in top_examples:
            # 加载训练集图像
            train_image_path = os.path.join(args.train_image_dir,
                                          f'COCO_train2014_{str(example["example_image_id"]).zfill(12)}.jpg')
            demo_image = Image.open(train_image_path).convert('RGB')
            demo_image_list.append(demo_image)
            demo_question_list.append(example['example_question'])
            demo_answer_list.append(example['example_most_common_answer'])

        # 准备查询图像
        query_image_path = os.path.join(args.val_image_dir,
                                      f'COCO_val2014_{str(item["query_image_id"]).zfill(12)}.jpg')
        query_image = Image.open(query_image_path).convert('RGB')

        # 使用OFv2生成答案
        output_text = ofv2_inference_n_promt(
            ofv2_model,
            device,
            query_image,
            item['query_question'],
            demo_image_list,
            demo_question_list,
            demo_answer_list,
            tokenizer,
            image_processor
        )

        # 提取答案
        extracted_answer = extract_short_answer(output_text)
        if extracted_answer in ["I don't know", "unanswerable", "unable to tell"]:
            extracted_answer = "unable to tell"

        # 保存结果
        result = {
            'question_id': int(query_id),
            'image_id': item['query_image_id'],
            'question': item['query_question'],
            'answer': extracted_answer,
            'gt_answers': item['query_answers'],  # 添加真实答案
            'method': method_results['metadata']['method'],
            'examples': [{
                'score': example['score'],
                'question': example['example_question'],
                'answer': example['example_most_common_answer'],
                'image_id': example['example_image_id']
            } for example in item['top_examples']]
        }
        results.append(result)

    return results

def main():
    args = parse_args()
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # 加载OFv2模型
    print("加载OFv2模型...")
    ofv2_model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path='ViT-L-14',
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.lm_tokenizer_path,
        cross_attn_every_n_layers=4,
        inference=True,
        precision='fp16',
        device=device,
        checkpoint_path=args.checkpoint_path,
    )

    # 加载各方法的匹配结果
    # methods = ['multimodal', 'text', 'image']
    methods = ['text', 'image']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for method in methods:
        try:
            print(f"\n评估{method}方法...")
            method_results = load_matching_results(args.analysis_dir, method, args)

            # 评估当前方法
            results = evaluate_method(method_results, args, ofv2_model, image_processor, tokenizer, device)
            all_results.extend(results)

            # 保存当前方法的结果
            output_file = os.path.join(args.output_dir, f"{method}_evaluation_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"{method}方法的结果已保存到: {output_file}")

            # 计算最终准确率
            final_accuracy = compute_vqa_accuracy(
                output_file,
                question_json_path=args.question_json_path,
                annotation_json_path=args.annotation_json_path,
            )

            print(f"最终预测准确率: {final_accuracy / 100:.2%}")

        except Exception as e:
            print(f"评估{method}方法时发生错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存所有方法的综合结果
    combined_output_file = os.path.join(args.output_dir, f"all_methods_evaluation_{timestamp}.json")
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"所有方法的综合结果已保存到: {combined_output_file}")

if __name__ == "__main__":
    main()