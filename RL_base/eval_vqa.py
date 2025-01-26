from policy_model_eval_okvqa import parse_args as eval_parse_args, load_trained_policy_model, get_okvqa_dataloader, \
    evaluate_policy_model
from policy_model_score_okvqa import main as score_main, parse_args as score_parse_args
from datetime import datetime
import os
import json
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_base.open_flamingo import create_model_and_transforms

def main():
    # 获取评估阶段的参数
    eval_args = eval_parse_args()
    # 获取评分阶段的参数
    score_args = score_parse_args()

    # 设置设备
    device = torch.device("cuda:" + eval_args.gpu if torch.cuda.is_available() else "cpu")

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

    # 第一阶段：运行评估过程
    print("第一阶段：开始示例检索...")

    # 加载模型
    policy_model, preprocess = load_trained_policy_model(eval_args.policy_model_checkpoint, device, eval_args)

    # 加载数据
    test_dataloader = get_okvqa_dataloader(
        eval_args.okvqa_test_questions_json_path,
        eval_args.okvqa_test_annotations_json_path,
        eval_args.test_image_dir,
        batch_size=eval_args.batch_size,
        num_workers=eval_args.num_workers,
        split="val"
    )

    train_dataloader = get_okvqa_dataloader(
        eval_args.okvqa_train_questions_json_path,
        eval_args.okvqa_train_annotations_json_path,
        eval_args.train_image_dir,
        batch_size=eval_args.batch_size,
        num_workers=eval_args.num_workers,
        split="trn"
    )

    # 执行评估
    best_examples = evaluate_policy_model(policy_model, test_dataloader, train_dataloader, eval_args, device,
                                          preprocess)

    # 保存评估结果
    results_dir = os.path.join(eval_args.output_path, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result_path = os.path.join(results_dir, f"retrieval_results_okvqa_{timestamp}.json")

    with open(eval_result_path, 'w', encoding='utf-8') as f:
        json.dump(best_examples, f, ensure_ascii=False, indent=2)

    print(f"示例检索完成，结果保存至: {eval_result_path}")

    # 第二阶段：运行评分过程
    print("\n第二阶段：开始模型评分...")

    # 更新评分参数
    score_args.input_file = eval_result_path
    score_args.output_file = os.path.join(results_dir, f"score_results_okvqa_{timestamp}.json")
    score_args.gpu = eval_args.gpu  # 使用相同的GPU
    score_args.shot_number = eval_args.shot_number  # 使用相同的示例数量


    # 执行评分
    score_main(score_args, ofv2_model, image_processor, tokenizer)

    print("整个流程执行完成！")


if __name__ == "__main__":
    main()