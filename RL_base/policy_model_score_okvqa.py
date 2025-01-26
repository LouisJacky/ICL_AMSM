import os
import json
import argparse
import torch
from PIL import Image
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL_base.open_flamingo import create_model_and_transforms
from RL_base.open_flamingo.inference import ofv2_inference_n_promt
from RL_base.open_flamingo.eval.vqa_metric import compute_vqa_accuracy
import re
from tqdm import tqdm
import math

from label_vqa import extract_short_answer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_path', type=str, default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str, default="/path/to/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str, default="/path/to/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--input_file', type=str, default='/path/to/AMSM/log/ofv2_base/END_OUTPUT_linear/okvqa/evaluation_results/retrieval_results_okvqa_20250123_213934.json')
    # parser.add_argument('--input_file', type=str,
    #                     default='/path/to/AMSM/RL_base/okvqa_test_train_matching.json')
    parser.add_argument('--output_file', type=str, default='ofv2_results.json')
    parser.add_argument('--val_image_dir', type=str, default="/path/to/datasets/ok_vqa/val2014")
    parser.add_argument('--train_image_dir', type=str, default="/path/to/datasets/ok_vqa/train2014")
    parser.add_argument('--question_json_path', type=str, default="/path/to/datasets/ok_vqa/OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument('--annotation_json_path', type=str, default="/path/to/datasets/ok_vqa/mscoco_val2014_annotations.json")
    parser.add_argument('--shot_number',
                        type=int,
                        default=4,
                        help='Prompt num')
    return parser.parse_args()

def main(args=None, ofv2_model=None, image_processor=None, tokenizer=None):
    if args is None:
        args = parse_args()

    # 设置设备
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    if ofv2_model is None:
        # 加载模型和处理器
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

    # 读取检索结果文件
    with open(args.input_file, 'r') as f:
        retrieval_results = json.load(f)

    results = []
    total_samples = len(retrieval_results)
    progress_step = math.ceil(total_samples * 0.05)  # 5%的样本数

    for i, (query_question_id, item) in enumerate(tqdm(retrieval_results.items(), desc="Processing samples")):
        query_image_id = item['query_image_id']
        query_question = item['query_question']
        top_examples = item['top_examples'][:args.shot_number]

        # 将最相似的样本排在最后
        top_examples.reverse()

        # 准备演示样本
        demo_image_list = []
        demo_question_list = []
        demo_answer_list = []

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
            f'COCO_val2014_{str(query_image_id).zfill(12)}.jpg')
        query_image = Image.open(query_image_path).convert('RGB')

        # 使用ofv2_inference_n_promt生成答案
        output_text = ofv2_inference_n_promt(
            ofv2_model,
            device,
            query_image,
            query_question,
            demo_image_list,
            demo_question_list,
            demo_answer_list,
            tokenizer,
            image_processor
        )

        # 从输出文本中提取答案
        extracted_answer = extract_short_answer(output_text)

        if extracted_answer in ["I don't know", "unanswerable", "unable to tell"]:
            extracted_answer = "unable to tell"

        # 保存结果
        result = {
            'question': query_question,
            'question_id': int(query_question_id),
            'image_id': query_image_id,
            'answer': extracted_answer
        }
        results.append(result)

        # 每完成5%的样本,进行一次临时的准确率计算
        if (i + 1) % progress_step == 0 or i == total_samples - 1:
            # 保存当前结果
            temp_output_file = f"temp_results_{i+1}.json"
            with open(temp_output_file, 'w') as f:
                json.dump(results, f, indent=2)

            # 计算临时准确率
            temp_accuracy = compute_vqa_accuracy(
                temp_output_file,
                question_json_path=args.question_json_path,
                annotation_json_path=args.annotation_json_path,
            )
            print(f"已处理 {i+1}/{total_samples} 个样本。当前准确率: {temp_accuracy/100:.2%}")

    # 将最终结果保存到新的JSON文件
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 计算最终准确率
    final_accuracy = compute_vqa_accuracy(
        args.output_file,
        question_json_path=args.question_json_path,
        annotation_json_path=args.annotation_json_path,
    )

    print(f"最终预测准确率: {final_accuracy/100:.2%}")

if __name__ == "__main__":
    main()