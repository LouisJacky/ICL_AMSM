import json
import random
from tqdm import tqdm
from collections import defaultdict

k = 32

# 计算方差并筛选样本
def calculate_variance(examples):
    labels = [example['label'] for example in examples]
    mean = sum(labels) / len(labels)
    variance = sum((x - mean) ** 2 for x in labels) / len(labels)
    return variance
# 打印统计信息
def print_stats():
    query_count = len(new_dataset)
    total_examples = sum(len(item['examples']) for item in new_dataset)

    print("\n数据集统计:")
    print(f"查询样本总数: {query_count}")
    print(f"示例样本总数: {total_examples}")
    print(f"平均每个查询的示例数: {total_examples / query_count:.2f}")

# 读取标签对文件
with open(f'/data16tb/ljq/Code/ICL_diversity_ofv3/RL_base/vizwiz_label_confidence_{k}.json', 'r') as f:
    pair_data = json.load(f)

# 读取问题和答案文件
with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/clustered_train_questions_vqa_format.json',
          'r') as f:
    questions_data = json.load(f)
with open(
        '/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/sampled/sampled_train_annotations_vqa_format.json',
        'r') as f:
    answers_data = json.load(f)

# 创建question_id到image_id的映射
question_to_image_id = {q['question_id']: q['image_id'] for q in questions_data['questions']}

# 创建查找字典 - 使用question_id作为键
questions_dict = {q['question_id']: q['question'] for q in questions_data['questions']}
answers_dict = {a['question_id']: a['answers'] for a in answers_data['annotations']}

# 创建新的数据集列表和查询示例字典
new_dataset = []
query_examples = defaultdict(list)

# 修改数据处理逻辑
for pair in tqdm(pair_data['samples'], desc="收集示例"):
    query_question_id = pair['query_question_id']
    example_question_id = pair['example_question_id']

    # 获取对应的image_id
    query_image_id = question_to_image_id.get(query_question_id)
    example_image_id = question_to_image_id.get(example_question_id)

    if query_image_id and example_image_id:  # 确保能找到对应的image_id
        query_examples[query_image_id].append({
            'example_image': example_image_id,
            'example_question': questions_dict.get(example_question_id, ""),
            'example_answers': [a["answer"] for a in answers_dict.get(example_question_id, [])],
            'label': pair['confidence_score'],
            'similarity_score': pair['similarity_score']  # 添加相似度分数
        })

# # 对每个查询的示例进行筛选，只保留相似度最高的8个
# for query_id in tqdm(query_examples, desc="筛选示例"):
#     examples = query_examples[query_id]
#     # 按相似度分数降序排序
#     sorted_examples = sorted(
#         examples,
#         key=lambda x: x['similarity_score'],
#         reverse=True
#     )
#     # 选取前8个（最高相似度）
#     query_examples[query_id] = sorted_examples[:8]

# # 存储所有query的方差
# variance_dict = {}
# for query_id, examples in tqdm(query_examples.items(), desc="计算方差"):
#     if examples:
#         variance = calculate_variance(examples)
#         variance_dict[query_id] = variance

# # 按方差降序排序
# sorted_queries = sorted(variance_dict.items(), key=lambda x: x[1], reverse=True)
#
# # 选择方差最大的前top_ratio样本
# top_ratio = 0.95  # 可以调整这个比例
# selected_queries = set([query_id for query_id, _ in sorted_queries[:int(len(sorted_queries) * top_ratio)]])

# 构建新的数据集
for query_image_id, examples in tqdm(query_examples.items(), desc="构建数据集"):
    if examples:
    # if query_image_id in selected_queries and examples:
        new_dataset.append({
            'query_image': query_image_id,
            'query_question': questions_dict.get(
                next(qid for qid, iid in question_to_image_id.items() if iid == query_image_id), ""),
            'query_answers': [a["answer"] for a in answers_dict.get(
                next(qid for qid, iid in question_to_image_id.items() if iid == query_image_id), [])],
            'examples': examples
        })

# 随机打乱数据集
random.shuffle(new_dataset)

print_stats()

# 保存处理后的数据集
output_file = '/data16tb/ljq/Code/ICL_diversity_ofv3/data/vizwiz_labeled_confidence.json'
with open(output_file, 'w') as f:
    json.dump(new_dataset, f, indent=2)

print(f"已将数据集保存到 {output_file}")

