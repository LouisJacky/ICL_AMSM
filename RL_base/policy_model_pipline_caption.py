from policy_model_eval_caption import parse_args as eval_parse_args, load_trained_policy_model
from policy_model_eval_caption import COCOCaptionDataset, evaluate_policy_model, custom_collate
from policy_model_score_caption_batch import main as score_main, parse_args as score_parse_args
from datetime import datetime
import os
import json
import torch
from torch.utils.data import DataLoader
from open_flamingo import create_model_and_transforms
# from policy_model_label_caption_cider import COCOCaptionDataset, custom_collate

def main():
    # 获取评估阶段的参数
    eval_args = eval_parse_args()
    # 获取评分阶段的参数
    score_args = score_parse_args()

    # 设置设备
    device = torch.device("cuda:" + eval_args.gpu if torch.cuda.is_available() else "cpu")

    # 加载OpenFlamingo模型（用于第二阶段）
    print("加载OpenFlamingo模型...")
    ofv2_model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path='ViT-L-14',
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=score_args.lm_path,
        tokenizer_path=score_args.lm_tokenizer_path,
        cross_attn_every_n_layers=4,
        inference=True,
        precision='fp16',
        device=device,
        checkpoint_path=score_args.checkpoint_path,
    )

    # 第一阶段：运行示例检索过程
    print("第一阶段：开始示例检索...")

    # 加载策略模型
    policy_model, preprocess = load_trained_policy_model(
        eval_args.policy_model_checkpoint,
        device,
        eval_args
    )

    # 加载数据集
    print("加载数据集...")
    # 创建数据加载器
    val_dataset = COCOCaptionDataset(eval_args.val_json_file, eval_args.val_image_dir, split="val")
    train_dataset = COCOCaptionDataset(eval_args.train_json_file, eval_args.train_image_dir, split="train")

    val_dataloader = DataLoader(val_dataset, batch_size=eval_args.batch_size,
                                shuffle=False, num_workers=eval_args.num_workers,
                                collate_fn=custom_collate)
    train_dataloader = DataLoader(train_dataset, batch_size=eval_args.batch_size,
                                  shuffle=False, num_workers=eval_args.num_workers,
                                  collate_fn=custom_collate)

    # 执行示例检索评估
    best_examples = evaluate_policy_model(
        policy_model,
        val_dataloader,
        train_dataloader,
        eval_args,
        device,
        preprocess
    )

    # 保存检索结果
    results_dir = os.path.join(eval_args.output_path, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result_path = os.path.join(results_dir, f"best_examples_{timestamp}.json")

    with open(eval_result_path, 'w', encoding='utf-8') as f:
        json.dump(best_examples, f, ensure_ascii=False, indent=2)

    print(f"示例检索完成，结果保存至: {eval_result_path}")

    # 第二阶段：运行描述生成和评分过程
    print("\n第二阶段：开始描述生成和评分...")

    # 更新评分参数
    score_args.input_file = eval_result_path
    score_args.output_file = os.path.join(results_dir, f"caption_results_{timestamp}.json")
    score_args.gpu = eval_args.gpu  # 使用相同的GPU
    score_args.shot_number = eval_args.shot_number  # 使用相同的示例数量

    # 执行描述生成和评分
    score_main(score_args, ofv2_model, image_processor, tokenizer)

    print("整个流程执行完成！")
    print(f"最终结果保存在: {score_args.output_file}")

if __name__ == "__main__":
    main()