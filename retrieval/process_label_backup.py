import json
import random
import itertools
from tqdm import tqdm

# 读取原始JSON文件
# with open('/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/vizwiz_label.json', 'r') as f:
with open('/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/vizwiz_label_1k_fixed.json', 'r') as f:
    data = json.load(f)

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

# 创建问题和答案的查找字典
questions_dict = {q['image_id']: q['question'] for q in questions_data['questions']}
answers_dict = {a['image_id']: a['answers'] for a in answers_data['annotations']}

# 创建一个新的列表来存储重构后的样本
new_dataset = []

# 获取所有图像ID
all_image_ids = list(set(list(data.keys()) + [img for sublist in data.values() for img in sublist]))

# 创建一个集合来存储所有正样本组合
positive_pairs = set()

# 创建一个字典来存储每个查询图像的正样本和负样本
query_samples = {img_id: {'positive': [], 'negative': []} for img_id in all_image_ids}

# 处理正样本
for query_image, example_images in tqdm(data.items(), desc="处理正样本"):
    query_question = questions_dict.get(query_image, "")
    query_answers = answers_dict.get(query_image, [])
    query_answers = [query_answer["answer"] for query_answer in query_answers]

    for example_image in example_images:
        if example_image != query_image:  # 确保示例不是查询自身
            example_question = questions_dict.get(example_image, "")
            example_answers = answers_dict.get(example_image, [])
            example_answers = [example_answer["answer"] for example_answer in example_answers]

            query_samples[query_image]['positive'].append({
                'example_image': example_image,
                'example_question': example_question,
                'example_answers': example_answers,
            })

# 创建所有可能的图像对组合
all_possible_pairs = set((q, e) for q in all_image_ids for e in all_image_ids if q != e)
positive_pairs = set((q, e) for q, examples in data.items() for e in examples if q != e)
negative_pairs = all_possible_pairs - positive_pairs

# 处理负样本
for query_image, example_image in tqdm(negative_pairs, desc="处理负样本"):
    example_question = questions_dict.get(example_image, "")
    example_answers = answers_dict.get(example_image, [])
    example_answers = [example_answer["answer"] for example_answer in example_answers]

    query_samples[query_image]['negative'].append({
        'example_image': example_image,
        'example_question': example_question,
        'example_answers': example_answers,
    })

# 平衡每个查询的正负样本
new_dataset = []
positive_samples_num = 0
negative_samples_num = 0
for query_image, samples in tqdm(query_samples.items(), desc="平衡每个查询的正负样本"):
    query_question = questions_dict.get(query_image, "")
    query_answers = answers_dict.get(query_image, [])
    query_answers = [query_answer["answer"] for query_answer in query_answers]

    num_positive = len(samples['positive'])
    num_negative = len(samples['negative'])
    num_samples = min(num_positive, num_negative)

    positive_samples = random.sample(samples['positive'], num_samples)
    negative_samples = random.sample(samples['negative'], num_samples)

    for example in positive_samples:
        new_dataset.append({
            'query_image': query_image,
            'query_question': query_question,
            'query_answers': query_answers,
            'example_image': example['example_image'],
            'example_question': example['example_question'],
            'example_answers': example['example_answers'],
            'label': 1  # 正样本标记
        })
        positive_samples_num += 1

    for example in negative_samples:
        new_dataset.append({
            'query_image': query_image,
            'query_question': query_question,
            'query_answers': query_answers,
            'example_image': example['example_image'],
            'example_question': example['example_question'],
            'example_answers': example['example_answers'],
            'label': 0  # 负样本标记
        })
        negative_samples_num += 1

# 随机打乱new_dataset中的样本顺序
random.shuffle(new_dataset)

# 打印新数据集的前几个样本以验证
print(json.dumps(new_dataset[:4], indent=2))

# 将打乱顺序后的数据集保存为新的JSON文件
output_file = '/data16tb/ljq/Code/ICL_diversity_ofv3/data/vizwiz_labeled_new_dataset_1k.json'
with open(output_file, 'w') as f:
    json.dump(new_dataset, f, indent=10)

print(f"已将随机打乱顺序后的数据集（包含正负样本）保存到 {output_file}")
# print(f"总样本数：{len(new_dataset)}，其中正样本数：{len(positive_pairs)}，负样本数：{len(new_dataset) - len(positive_pairs)}")
print(f"总样本数：{len(new_dataset)}，其中原正样本数：{len(positive_pairs)}，保留正样本数：{positive_samples_num}，添加负样本数：{negative_samples_num}")

