from RL_base.open_flamingo import create_model_and_transforms
import torch
from PIL import Image
from PIL import ImageFilter
import requests
import itertools

from typing import List, Dict
import torch.nn.functional as F
from open_flamingo.eval.classification_utils import IMAGENET_CLASSNAMES
import numpy as np

class PATH:
    lm_path = "path for mpt-7b"
    lm_tokenizer_path = "path for mpt-7b"
    checkpoint_path = "path for openflamingo v2 checkpoint.pt"


def ofv2_inference(flamingo, device, query_image, query_question, demo_image, demo_question, demo_answer, tokenizer, image_processor):
    tokenizer.padding_side = "left"
    lang_x = tokenizer(
        [f"<image>Question: {demo_question} Answer: {demo_answer}. <|endofchunk|><image>Question: {query_question} Short answer:"],
        return_tensors="pt",
    )
    vision_x = [image_processor(demo_image).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    # load data to gpus
    vision_x = vision_x.to(device).half()
    # print(vision_x.device)
    input_ids=lang_x["input_ids"]
    attention_mask = lang_x["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    generated_text = flamingo.generate(
        vision_x=vision_x,
        lang_x=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        num_beams=3,
    )
    return tokenizer.decode(generated_text[0])



def ofv2_inference_n_promt(flamingo, device, query_image, query_question, demo_image_list, demo_question_list, demo_answer_list, tokenizer, image_processor):
    tokenizer.padding_side = "left"
    text = ""
    for i in range(len(demo_question_list)):
        text_0 = f"<image>Question: {demo_question_list[i]} Answer: {demo_answer_list[i]}. <|endofchunk|>"
        text = text + text_0
    text = text + f"<image>Question: {query_question} Short answer:"

    lang_x = tokenizer(
        [text],
        return_tensors="pt",
    )
    vision_x = []
    for i in range(len(demo_image_list)):
        vision_x.append(image_processor(demo_image_list[i]).unsqueeze(0))
    vision_x.append(image_processor(query_image).unsqueeze(0))

    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    # load data to gpus
    vision_x = vision_x.to(device).half()
    # print(vision_x.device)
    input_ids=lang_x["input_ids"]
    attention_mask = lang_x["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    generated_text = flamingo.generate(
        vision_x=vision_x,
        lang_x=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        num_beams=3,
    )
    return tokenizer.decode(generated_text[0])


def ofv2_classification(
        flamingo,
        device,
        query_image,
        query_labels,  # 以", "分隔的标签
        demo_images: List,
        demo_labels: List,  # 每个元素是以", "分隔的同义词字符串
        tokenizer,
        image_processor,
):
    """
    计算查询图像对应每个可能标签的置信度分数

    Returns:
        Dict: 包含所有预测标签及其置信度分数
    """
    debug = False
    # 构建演示文本
    text = ""
    for i in range(len(demo_images)):
        primary_label = demo_labels[i].split(", ")[0]
        text += f"<image>Output:{primary_label}<|endofchunk|>"

    # 处理图像
    vision_x = []
    for demo_image in demo_images:
        vision_x.append(image_processor(demo_image).unsqueeze(0))
    vision_x.append(image_processor(query_image).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    vision_x = vision_x.to(device).half()

    # 计算每个标签的置信度
    label_scores = []
    # 根据输入类型处理查询标签
    query_label_list = query_labels.split(", ") if isinstance(query_labels, str) else query_labels

    for label in query_label_list:
        text_label =  text + f"<image>Output:{label}"
        # 获取提示文本的tokens
        prompt_tokens = tokenizer(text_label, return_tensors="pt").to(device)
        # 获取目标标签的tokens
        target_tokens = tokenizer(label, return_tensors="pt").input_ids.to(device)
        target_length = target_tokens.size(1)

        with torch.no_grad():
            outputs = flamingo(
                vision_x=vision_x,
                lang_x=prompt_tokens.input_ids,
                attention_mask=prompt_tokens.attention_mask,
                labels=None,
            )

            logits = outputs.logits[:, -target_length - 1:-1, :]
            scores = F.log_softmax(logits, dim=-1)
            probs = F.softmax(scores, dim=-1)

            if debug:
                # 添加调试信息
                print("\nDebug Information:")
                print(f"Target answer: {label}")
                print(f"Target tokens: {[tokenizer.decode([id.item()]) for id in target_tokens[0]]}")

            # 计算每个token的置信度
            target_probs = []
            for i in range(target_length):
                token_probs = probs[0, i]
                target_token_id = target_tokens[0, i].item()
                target_prob = max(token_probs[target_token_id].item(), 1e-10)
                target_probs.append(target_prob)

                # 打印top-k预测
                if debug:
                    top_probs, top_indices = token_probs.topk(5)
                    print(f"\nPosition {i}:")
                    print(f"Target token: {tokenizer.decode([target_token_id])}")
                    print(f"Target probability: {token_probs[target_token_id].item():.4f}")
                    print("Top 5 predictions:")
                    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                        print(f"{tokenizer.decode([idx])}: {prob:.4f}")

            # 计算平均置信度
            avg_confidence = sum(target_probs) / len(target_probs) if target_probs else 0.0
            # # 计算几何平均置信度
            # avg_confidence = np.prod(target_probs) ** (1.0 / len(target_probs)) if target_probs else 0.0
            label_scores.append({"label": label, "confidence": avg_confidence})

    # 按置信度排序
    label_scores.sort(key=lambda x: x["confidence"], reverse=True)

    return {"predictions": label_scores}

