import os
import json
import argparse
import torch
from PIL import Image
from open_flamingo import create_model_and_transforms
from tqdm import tqdm
import math
from datetime import datetime
from collections import defaultdict
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_base.cococaption.pycocoevalcap.cider.cider import Cider


def parse_args():
    parser = argparse.ArgumentParser()
    # OFv2模型参数
    parser.add_argument('--lm_path', type=str,
                        default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--lm_tokenizer_path', type=str,
                        default="/data16tb/ljq/checkpoints/mpt-7b")
    parser.add_argument('--checkpoint_path', type=str,
                        default="/data16tb/ljq/checkpoints/ofv2/checkpoint.pt")
    parser.add_argument('--gpu', type=str, default='4')

    # 数据路径
    parser.add_argument('--val_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa/val2014')
    parser.add_argument('--train_image_dir', type=str,
                        default='/data16tb/ljq/datasets/ok_vqa/train2014')

    # 分析结果路径
    parser.add_argument('--analysis_dir', type=str, default="caption_analysis_results")
    parser.add_argument('--output_dir', type=str, default="caption_evaluation_results")

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shot_number', type=int, default=1)
    # 添加采样相关参数
    parser.add_argument('--sample_size', type=int, default=None,
                        help='要评估的样本数量，不设置则评估所有样本')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机采样的种子')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def load_matching_results(analysis_dir, method):
    """加载指定方法的匹配结果"""
    matching_files = [f for f in os.listdir(analysis_dir) if f.startswith(f'{method}_matching_')]
    if not matching_files:
        raise FileNotFoundError(f"未找到{method}方法的匹配结果文件")

    # 使用最新的结果文件
    latest_file = max(matching_files, key=lambda x: os.path.getctime(os.path.join(analysis_dir, x)))
    with open(os.path.join(analysis_dir, latest_file), 'r') as f:
        return json.load(f)

def extract_caption(text, prompt_num=1):
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


def compute_cider_scores(gts, res):
    """计算CIDEr分数"""
    scorer = Cider()
    _, scores = scorer.compute_score(gts, res)
    return scores


def evaluate_method(method_results, args, ofv2_model, image_processor, tokenizer, device, selected_sample_ids):
    """评估特定方法的性能"""
    results = []
    gts = defaultdict(list)
    res = {}

    # 获取所有样本ID列表
    all_sample_ids = list(method_results['results'].keys())
    total_samples = len(all_sample_ids)

    # 如果指定了sample_size，进行随机采样
    if args.sample_size is not None and args.sample_size < total_samples:
        # sampled_ids = torch.randperm(total_samples)[:args.sample_size].tolist()
        # selected_sample_ids = [all_sample_ids[i] for i in sampled_ids]
        total_samples = args.sample_size
    else:
        selected_sample_ids = all_sample_ids
    batch_size = args.batch_size


    # # 使用tqdm包装批次迭代
    # for i in tqdm(range(0, total_samples, batch_size), desc=f"评估{method_results['metadata']['method']}方法"):
    #     batch_end = min(i + batch_size, total_samples)
    #     batch_items = list(method_results['results'].values())[i:batch_end]
    # 使用选定的样本ID进行评估
    # 使用选定的样本ID进行评估
    for i in tqdm(range(0, len(selected_sample_ids), batch_size), desc=f"评估{method_results['metadata']['method']}方法"):
        batch_end = min(i + batch_size, len(selected_sample_ids))
        batch_ids = selected_sample_ids[i:batch_end]
        batch_items = [method_results['results'][id] for id in batch_ids]

        batch_query_images = []
        batch_demo_images = []
        batch_demo_captions = []

        # 准备批处理数据
        for item in batch_items:
            # 准备查询图像
            query_image_path = os.path.join(args.val_image_dir, item['query_file_name'])
            query_image = Image.open(query_image_path).convert('RGB')
            batch_query_images.append(query_image)

            # 准备演示样本
            demo_images_single = []
            demo_captions_single = []
            for example in item['top_examples'][:args.shot_number]:
                demo_image_path = os.path.join(args.train_image_dir, example['example_file_name'])
                demo_image = Image.open(demo_image_path).convert('RGB')
                demo_images_single.append(demo_image)
                demo_captions_single.append(example['example_captions'][0])

            batch_demo_images.append(demo_images_single)
            batch_demo_captions.append(demo_captions_single)

        # 批量生成caption
        generated_captions = ofv2_inference_caption(
            ofv2_model,
            device,
            batch_query_images,
            batch_demo_images,
            batch_demo_captions,
            tokenizer,
            image_processor
        )

        # 处理生成的caption
        for j, caption in enumerate(generated_captions):
            current_idx = i + j
            if current_idx >= total_samples:
                break

            item = batch_items[j]

            # 提取生成的caption
            generated_caption = extract_caption(caption, args.shot_number)

            # 保存结果
            result = {
                'query_image_id': item['query_image_id'],
                'query_file_name': item['query_file_name'],
                'ground_truth_captions': item['query_captions'],
                'generated_caption': generated_caption,
                'examples': item['top_examples'][:args.shot_number]
            }
            results.append(result)

            # 准备计算CIDEr分数
            gts[current_idx] = item['query_captions']
            res[current_idx] = [generated_caption]

            # 检查是否需要保存临时结果
            # current_count = len(results)
            # if (current_count - last_save_count >= save_interval) or (current_count == total_samples):
            #     temp_scores = compute_cider_scores(gts, res)
            #     avg_score = sum(temp_scores) / len(temp_scores)
            #
            #     tqdm.write(f"\n已处理 {current_count}/{total_samples} 个样本 "
            #                f"({(current_count / total_samples) * 100:.1f}%). "
            #                f"当前平均CIDEr分数: {avg_score:.4f}")
            #
            #     last_save_count = current_count

    # 计算最终CIDEr分数
    final_scores = compute_cider_scores(gts, res)
    avg_final_score = sum(final_scores) / len(final_scores)

    # 添加分数到结果中
    for i, result in enumerate(results):
        result['cider_score'] = float(final_scores[i])

    return {
        'method': method_results['metadata']['method'],
        'results': results,
        'average_cider_score': float(avg_final_score),
        'total_samples': total_samples
    }


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


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

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
        checkpoint_path=args.checkpoint_path
    )

    # 首先加载一个方法的结果来获取所有可用的样本ID
    method_results = load_matching_results(args.analysis_dir, 'multimodal')
    all_sample_ids = list(method_results['results'].keys())
    total_samples = len(all_sample_ids)

    # 进行随机采样，确保所有方法使用相同的样本
    if args.sample_size is not None and args.sample_size < total_samples:
        sampled_indices = torch.randperm(total_samples)[:args.sample_size].tolist()
        selected_sample_ids = [all_sample_ids[i] for i in sampled_indices]
        print(f"随机选择了 {args.sample_size} 个样本进行评估")
    else:
        selected_sample_ids = all_sample_ids
        print(f"使用全部 {total_samples} 个样本进行评估")

    # 评估每种方法
    methods = ['multimodal', 'image', 'random']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for method in methods:
        try:
            print(f"\n评估{method}方法...")
            method_results = load_matching_results(args.analysis_dir, method)

            # # 评估当前方法
            # results = evaluate_method(method_results, args, ofv2_model,
            #                           image_processor, tokenizer, device)
            # 评估当前方法，传入选定的样本ID
            results = evaluate_method(method_results, args, ofv2_model,
                                      image_processor, tokenizer, device,
                                      selected_sample_ids)  # 新增参数

            all_results.append(results)

            # 保存当前方法的结果
            output_file = os.path.join(args.output_dir, f"{method}_caption_eval_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"{method}方法的结果已保存到: {output_file}")

        except Exception as e:
            print(f"评估{method}方法时发生错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存所有方法的综合结果
    combined_output_file = os.path.join(args.output_dir, f"all_methods_caption_eval_{timestamp}.json")
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"所有方法的综合结果已保存到: {combined_output_file}")


if __name__ == "__main__":
    main()