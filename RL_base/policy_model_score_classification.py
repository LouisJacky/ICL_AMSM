import os
import json
import argparse
import torch
from PIL import Image
from open_flamingo import create_model_and_transforms
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from open_flamingo.inference import ofv2_classification
from retrieval.process_label_caption_cider import extract_short_answer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_path', type=str,
                        default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str,
                        default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str,
                        default="/path/to/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--input_file', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear/tiny_imagenet/tiny_imagenet_classification_matches_random.json')
    # parser.add_argument('--input_file', type=str,
    #                     default='/path/to/AMSM/log/ofv2_base/END_OUTPUT_linear/tiny_imagenet/tiny_imagenet_classification_matches_si.json')
    parser.add_argument('--output_file', type=str,
                        default='classification_results.json')
    parser.add_argument('--root_dir', type=str,
                        default='/path/to/datasets/tiny-imagenet-200')
    parser.add_argument('--shot_number', type=int, default=4)
    return parser.parse_args()

def compute_accuracy(results):
    total_count = 0
    total_score = 0
    correct_count = 0
    for result in results:
        total_count += 1
        total_score += result['confidence_score']
        if result['correct']:
            correct_count += 1.0

    mean_score = total_score / total_count
    correct_rate = correct_count / total_count
    return mean_score, correct_rate

def load_class_descriptions(root_dir):
    """加载类别描述文本和构建同义词映射"""
    class_descriptions = {}
    class_synonyms = {}

    with open(os.path.join(root_dir, 'words.txt'), 'r') as f:
        for line in f:
            class_id, description = line.strip().split('\t')
            class_descriptions[class_id] = description
            # 将描述分割成同义词列表
            synonyms = [s.strip() for s in description.split(', ')]
            class_synonyms[class_id] = synonyms

    return class_descriptions, class_synonyms


def main(args=None, ofv2_model=None, image_processor=None, tokenizer=None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    # 加载类别描述和同义词
    class_descriptions, class_synonyms = load_class_descriptions(args.root_dir)
    # all_possible_labels = []
    # for synonyms in class_synonyms.values():
    #     for synonym in synonyms:
    #         all_possible_labels.append(synonym)

    # 加载模型和处理器
    if ofv2_model is None:
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
    ofv2_model.eval()

    # 读取评估结果文件
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    results = []
    total_samples = len(data)
    progress_step = max(1, int(total_samples * 0.05))  # 计算5%的样本数

    for i, item in enumerate(tqdm(data, desc="处理样本")):
        query_label = item['query_label']
        best_examples = item['best_examples']
        # 将最相似的样本排在最后
        best_examples.reverse()

        # 准备演示样本
        demo_image_list = []
        # demo_images = []
        demo_label_list = []

        for example in best_examples[:args.shot_number]:
            image_path = os.path.join(args.root_dir, 'train', example['example_file_name'])
            demo_image = Image.open(image_path).convert('RGB')
            # demo_images.append(example['example_file_name'])
            demo_image_list.append(demo_image)
            demo_label_list.append(example['example_label'])

        # 准备查询图像
        query_image_path = os.path.join(args.root_dir, 'val', item['query_file_name'])
        query_image = Image.open(query_image_path).convert('RGB')


        # 生成预测
        confidence_score, correct = ofv2_inference_classification(
            ofv2_model,
            device,
            query_image,
            query_label,
            demo_image_list,
            demo_label_list,
            tokenizer,
            image_processor,
            class_descriptions,
        )

        # 保存结果
        result = {
            'query_file_name': item['query_file_name'],
            # 'demo_images': demo_images,
            'confidence_score': confidence_score,
            'correct': correct,

        }
        results.append(result)

        # 每处理5%的样本，保存临时结果并计算准确率
        if (i + 1) % progress_step == 0 or i == total_samples - 1:
            # 计算当前准确率
            mean_score, current_accuracy = compute_accuracy(results)

            # 准备临时输出数据
            temp_output_data = {
                'results': results,
                'current_accuracy': current_accuracy,
                'current_mean_score':mean_score,
                'processed_samples': i + 1,
                'total_samples': total_samples
            }

            # 保存临时结果
            temp_output_file = f"temp_classification_results_{i + 1}.json"
            with open(temp_output_file, 'w') as f:
                json.dump(temp_output_data, f, indent=2)

            print(f"处理了 {i + 1}/{total_samples} 个样本. 当前准确率: {current_accuracy:.4f}")

    # 计算最终准确率
    final_mean_score, final_accuracy = compute_accuracy(results)

    # 保存最终结果
    output_data = {
        'results': results,
        'final_accuracy': final_accuracy,
        'final_mean_score': final_mean_score,
        'total_samples': total_samples
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"最终准确率: {final_accuracy:.4f}")
    print(f"最终平均置信度分数: {final_mean_score:.4f}")
    print(f"总样本数: {total_samples}")


def ofv2_inference_classification(ofv2_model,
            device,
            query_image,
            query_labels,
            demo_image_list,
            demo_label_list,
            tokenizer,
            image_processor,
            class_descriptions,
                                  ):
    all_possible_labels = ''
    example_images = demo_image_list
    query_texts = class_descriptions.get(query_labels,'')
    all_possible_labels += query_texts + ', '
    example_texts = []
    for demo_label in demo_label_list:
        demo_texts = class_descriptions.get(demo_label,'')
        example_texts.append(demo_texts)
        all_possible_labels += demo_texts + ', '

    # 使用 ofv2_classification 计算置信度分数
    results = ofv2_classification(
        ofv2_model,
        device,
        query_image,
        all_possible_labels,
        example_images,
        example_texts,
        tokenizer,
        image_processor,
    )
    correct = False
    top_labels = []
    top_results = results["predictions"][:1]
    for top_result in top_results:
        top_labels.append(top_result["label"])

    query_texts_list = query_texts.split(', ')

    for label in query_texts_list:
        if label in top_labels:
            correct = True

    confidence = 0
    all_results = results["predictions"][:]
    for result in all_results:
        if result["label"] in query_texts_list:
            confidence = result['confidence']
            break

    # 返回置信度最高的预测结果
    return confidence, correct

if __name__ == "__main__":
    main()