import json
import random
import itertools

# 读取原始JSON文件
# with open('/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/vizwiz_label.json', 'r') as f:
with open('/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/vizwiz_label.json', 'r') as f:
    sampled_data = json.load(f)

# 读取问题JSON文件
with open(
        '/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/clustered_train_questions_vqa_format.json',
        'r') as f:
    questions_data = json.load(f)

# 读取答案JSON文件
with open(
        '/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/sampled/sampled_train_annotations_vqa_format.json',
        'r') as f:
    answers_data = json.load(f)

# 假设我们要取出的样本比例
sample_ratio = 1  # 这里设置为50%，您可以根据需要调整

# 获取所有query_image的列表
all_query_images = list(sampled_data.keys())

# 随机打乱query_image列表
random.shuffle(all_query_images)

# 计算要保留的样本数量
num_samples_to_keep = int(len(all_query_images) * sample_ratio)

# 只保留指定数量的query_image
selected_query_images = all_query_images[:num_samples_to_keep]

# 创建问题和答案的查找字典
questions_dict = {q['image_id']: q['question'] for q in questions_data['questions']}
answers_dict = {a['image_id']: a['answers'] for a in answers_data['annotations']}

# 创建一个新的列表来存储重构后的样本对
new_dataset = []

# 为每个选中的查询样本创建正负样本对
for query_image in selected_query_images:
    samples = sampled_data[query_image]
    positive_samples = samples['positive']
    negative_samples = samples['negative']

    # 如果正样本或负样本为空，则跳过此查询图像
    if not positive_samples or not negative_samples:
        continue

    # 确定可以创建的样本对数量，取较大值
    num_pairs = min(len(positive_samples), len(negative_samples))

    # 随机打乱正负样本列表
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # 使用itertools.cycle创建无限循环的迭代器
    positive_cycle = itertools.cycle(positive_samples)
    negative_cycle = itertools.cycle(negative_samples)

    query_question = questions_dict.get(query_image, "")
    query_answers = answers_dict.get(query_image, [])
    query_answers = [query_answer["answer"] for query_answer in query_answers]

    # 创建样本对
    for _ in range(num_pairs):
        positive_image = next(positive_cycle)
        negative_image = next(negative_cycle)

        positive_question = questions_dict.get(positive_image, "")
        positive_answers = answers_dict.get(positive_image, [])
        positive_answers = [answer["answer"] for answer in positive_answers]

        negative_question = questions_dict.get(negative_image, "")
        negative_answers = answers_dict.get(negative_image, [])
        negative_answers = [answer["answer"] for answer in negative_answers]

        new_dataset.append({
            'query_image': query_image,
            'query_question': query_question,
            'query_answers': query_answers,
            'positive_image': positive_image,
            'positive_question': positive_question,
            'positive_answers': positive_answers,
            'negative_image': negative_image,
            'negative_question': negative_question,
            'negative_answers': negative_answers
        })

# 随机打乱new_dataset中的样本顺序
random.shuffle(new_dataset)

# 打印新数据集的前几个样本以验证
print(json.dumps(new_dataset[:2], indent=2))

# 将打乱顺序后的数据集保存为新的JSON文件
output_file = '/data16tb/ljq/Code/ICL_diversity_ofv3/data/vizwiz_labeled_paired_dataset.json'
with open(output_file, 'w') as f:
    json.dump(new_dataset, f, indent=2)

print(f"已将随机打乱顺序后的配对数据集保存到 {output_file}")
print(f"总样本对数：{len(new_dataset)}")

