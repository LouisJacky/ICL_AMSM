import os
import json
import argparse
import torch
from PIL import Image
from open_flamingo import create_model_and_transforms
from tqdm import tqdm
import math
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_base.cococaption.pycocotools.coco import COCO
from RL_base.cococaption.pycocoevalcap.cider.cider import Cider
from collections import defaultdict
from retrieval.process_label_caption_cider import extract_short_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_path', type=str,
                        default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str,
                        default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str,
                        default="/data16tb/ljq/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='2')
    # parser.add_argument('--input_file', type=str,
    #                     default='../log/ofv2_base/END_OUTPUT_linear/coco_caption/evaluation_results/best_examples_20250105_231937.json')
    parser.add_argument('--input_file', type=str,
                        default='/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/coco_caption_test_train_matching_si_8promt.json')
    parser.add_argument('--output_file', type=str,
                        default='caption_generation_results.json')
    parser.add_argument('--train_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa/train2014')
    parser.add_argument('--val_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa/val2014')
    parser.add_argument('--annotations_path', type=str,
                        default='/data16tb/ljq/datasets/coco_caption/annotations_captions_val2014.json')
    parser.add_argument('--shot_number', type=int, default=8)

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--output_root', type=str,
                        default='../log/ofv2_base/END_OUTPUT_linear')
    parser.add_argument('--BENCHMARK', default='coco_caption')
    args = parser.parse_args()
    args.output_path = os.path.join(args.output_root, args.BENCHMARK)
    os.makedirs(args.output_path, exist_ok=True)
    return args


def compute_cider_scores(gts, res):
    """计算CIDEr分数"""
    scorer = Cider()
    _, scores = scorer.compute_score(gts, res)
    return scores


def main(args=None, ofv2_model=None, image_processor=None, tokenizer=None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

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

    # 加载COCO标注
    print("加载COCO标注...")
    coco = COCO(args.annotations_path)

    # 读取评估结果文件
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # 创建临时结果和最终结果的保存目录
    evaluation_results_dir = os.path.join(args.output_path, 'evaluation_results')
    os.makedirs(evaluation_results_dir, exist_ok=True)

    results = []
    gts = defaultdict(list)
    res = {}

    total_samples = len(data)
    batch_size = args.batch_size
    # 计算5%对应的样本数
    save_interval = max(1, math.ceil(total_samples * 0.05))
    last_save_count = 0

    # 使用tqdm包装批次迭代
    for i in tqdm(range(0, total_samples, batch_size), desc="处理批次"):
        batch_data = data[i:i + batch_size]
        batch_query_images = []
        batch_demo_images = []
        batch_demo_captions = []

        # 准备批处理数据
        for item in batch_data:
            # 获取演示样本
            examples = item['best_examples'][:args.shot_number]
            examples.reverse()

            # 准备单个查询的演示样本
            demo_images_single = []
            demo_captions_single = []

            for example in examples:
                demo_image = Image.open(os.path.join(args.train_image_dir, example['example_file_name']))
                demo_images_single.append(demo_image)
                demo_captions_single.append(example['example_captions'][0])

            # 准备查询图像
            query_image = Image.open(os.path.join(args.val_image_dir, item['query_file_name']))

            batch_query_images.append(query_image)
            batch_demo_images.append(demo_images_single)
            batch_demo_captions.append(demo_captions_single)

        # 批量生成描述
        generated_captions = ofv2_inference_caption(
            ofv2_model,
            device,
            batch_query_images,
            batch_demo_images,
            batch_demo_captions,
            tokenizer,
            image_processor
        )

        # 处理生成的描述
        for j, caption in enumerate(generated_captions):
            current_idx = i + j
            if current_idx >= total_samples:
                break

            item = batch_data[j]
            query_image_id = item['query_image_id']

            # 提取生成的描述
            generated_caption = extract_short_answer(caption, prompt_num=args.shot_number)
            if not generated_caption:
                generated_caption = "no description"

            # 保存结果
            result = {
                'query_image_id': query_image_id,
                'query_file_name': item['query_file_name'],
                'ground_truth_captions': item['query_captions'],
                'generated_caption': generated_caption,
                'demo_captions': batch_demo_captions[j]
            }
            results.append(result)

            # 准备计算CIDEr分数
            gts[current_idx] = item['query_captions']
            res[current_idx] = [generated_caption]

            # 检查是否需要保存临时结果
            current_count = len(results)
            if (current_count - last_save_count >= save_interval) or (current_count == total_samples):
                # 暂停进度条
                tqdm.write("\n")  # 换行，避免与进度条冲突

                # 保存当前结果
                temp_output_file = os.path.join(
                    evaluation_results_dir,
                    f"temp_results_{current_count}.json"
                )

                # 计算临时CIDEr分数
                temp_scores = compute_cider_scores(gts, res)
                avg_score = sum(temp_scores) / len(temp_scores)

                # 创建包含分数的临时结果字典
                temp_output = {
                    'results': results,
                    'average_cider_score': float(avg_score),
                    'current_samples': current_count,
                    'total_samples': total_samples,
                    'batch_size': batch_size,
                }

                # 保存当前结果
                with open(temp_output_file, 'w') as f:
                    json.dump(temp_output, f, indent=2)

                # 使用tqdm.write而不是print，这样不会干扰进度条的显示
                tqdm.write(f"已处理 {current_count}/{total_samples} 个样本 ({(current_count / total_samples) * 100:.1f}%). "
                           f"当前平均CIDEr分数: {avg_score:.4f}")

                last_save_count = current_count

    # 计算最终CIDEr分数
    final_scores = compute_cider_scores(gts, res)
    avg_final_score = sum(final_scores) / len(final_scores)

    # 将分数添加到结果中
    for i, result in enumerate(results):
        result['cider_score'] = float(final_scores[i])

    # 准备最终输出
    final_output = {
        'results': results,
        'average_cider_score': float(avg_final_score),
        'total_samples': total_samples,
        'batch_size': batch_size,
    }

    # 生成最终结果文件名
    final_output_file = os.path.join(
        evaluation_results_dir,
        f"final_results.json"
    )

    # 保存最终结果
    with open(final_output_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n处理完成！")
    print(f"总样本数: {total_samples}")
    print(f"最终平均CIDEr分数: {avg_final_score:.4f}")
    print(f"结果已保存至: {final_output_file}")


def ofv2_inference_caption(model, device, query_images, demo_images_list,
                           demo_captions_list, tokenizer, image_processor):
    """执行批量caption生成"""
    model.eval()
    with torch.no_grad():
        batch_vision_x = []
        batch_prompt_texts = []

        # 处理每个查询样本
        for query_image, demo_images, demo_captions in zip(query_images, demo_images_list, demo_captions_list):
            # 处理图像
            vision_x = []
            for demo_img in demo_images:
                processed_img = image_processor(demo_img).unsqueeze(0)  # [1, C, H, W]
                vision_x.append(processed_img)

            # 添加查询图像
            query_processed = image_processor(query_image).unsqueeze(0)  # [1, C, H, W]
            vision_x.append(query_processed)

            # 将所有图像堆叠在一起 [T_img, C, H, W]
            vision_x = torch.cat(vision_x, dim=0)
            # 添加特征维度 F=1 [T_img, 1,  C, H, W] 添加特征维度 Batch=1 [1, T_img, 1,  C, H, W]
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)
            batch_vision_x.append(vision_x)

            # 构建提示文本
            prompt_text = ""
            for caption in demo_captions:
                prompt_text += f"<image>Output:{caption}<|endofchunk|>"
            prompt_text += "<image>Output:"
            batch_prompt_texts.append(prompt_text)

        # 批量处理图像 [B, 1, T_img, C, H, W]
        vision_x = torch.cat(batch_vision_x, dim=0)
        vision_x = vision_x.to(device).half()

        # 修改tokenizer设置，使用左侧填充
        tokenizer.padding_side = 'left'  # 设置为左侧填充
        # tokenizer.pad_token = tokenizer.eos_token  # 确保有正确的填充token

        # 批量处理文本
        lang_x = tokenizer(
            batch_prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            pad_to_multiple_of=8,  # 确保填充长度是8的倍数，可以提高效率
        ).to(device)

        # 批量生成描述
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        # 生成完成后恢复tokenizer的默认设置
        tokenizer.padding_side = 'right'

        output_texts = [tokenizer.decode(text, skip_special_tokens=True)
                        for text in generated_text]
        return output_texts


if __name__ == "__main__":
    main()