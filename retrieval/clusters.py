import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
import pickle

# 设置NLTK数据目录
nltk_data_dir = '/data16tb/ljq/nltk_data'
nltk.data.path.append(nltk_data_dir)

# 预处理函数
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# 读取和处理训练数据
print("正在读取训练数据...")
with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/sampled/sampled_train_questions_vqa_format.json', 'r') as f:
    train_data = json.load(f)

train_questions = [q['question'] for q in train_data['questions']]
print("正在预处理训练问题...")
preprocessed_train_questions = [preprocess(q) for q in tqdm(train_questions, desc="预处理")]

# 使用TF-IDF向量化
print("正在进行TF-IDF向量化...")
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(tqdm(preprocessed_train_questions, desc="向量化"))

# 使用K-means聚类
n_clusters = 1
print(f"正在使用K-means进行聚类 (n_clusters={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_train)

# 将聚类结果添加到训练数据中
print("正在将聚类结果添加到训练数据...")
for i, q in tqdm(enumerate(train_data['questions']), total=len(train_data['questions']), desc="添加聚类结果"):
    q['cluster'] = int(kmeans.labels_[i])

# 统计每个聚类的样本数量
cluster_counts = [0] * n_clusters
for q in train_data['questions']:
    cluster_counts[q['cluster']] += 1

# 输出每个聚类的样本数量
print("每个聚类的样本数量：")
for i, count in enumerate(cluster_counts):
    print(f"聚类 {i}: {count} 个样本")

# 统计每个聚类的样本数量和不重复图片数量
cluster_counts = [0] * n_clusters
cluster_unique_images = [set() for _ in range(n_clusters)]
for q in train_data['questions']:
    cluster = q['cluster']
    cluster_counts[cluster] += 1
    cluster_unique_images[cluster].add(q['image_id'])

# 输出每个聚类的样本数量和不重复图片数量
print("每个聚类的样本数量和不重复图片数量：")
for i, (count, unique_images) in enumerate(zip(cluster_counts, cluster_unique_images)):
    print(f"聚类 {i}: {count} 个样本, {len(unique_images)} 张不重复图片")

# 保存训练数据的聚类结果
print("正在保存训练数据的聚类结果...")
with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/clustered_train_questions_vqa_format.json', 'w') as f:
    json.dump(train_data, f, indent=2)

# # 保存模型和向量化器
# print("正在保存模型和向量化器...")
# with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/kmeans_model.pkl', 'wb') as f:
#     pickle.dump(kmeans, f)
# with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)

# 读取和处理验证数据
print("正在读取验证数据...")
with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/sampled/sampled_val_questions_vqa_format.json', 'r') as f:
    val_data = json.load(f)

val_questions = [q['question'] for q in val_data['questions']]
print("正在预处理验证问题...")
preprocessed_val_questions = [preprocess(q) for q in tqdm(val_questions, desc="预处理")]

# 对验证数据进行向量化和预测
print("正在对验证数据进行向量化和预测...")
X_val = vectorizer.transform(tqdm(preprocessed_val_questions, desc="向量化"))
val_predictions = kmeans.predict(X_val)

# 将预测结果添加到验证数据中
print("正在将预测结果添加到验证数据...")
for i, q in tqdm(enumerate(val_data['questions']), total=len(val_data['questions']), desc="添加预测结果"):
    q['cluster'] = int(val_predictions[i])

# 保存验证数据的预测结果
print("正在保存验证数据的预测结果...")
with open('/data16tb/ljq/Code/OFv2_ICL_VQA/open_flamingo/eval/data/vizwiz/clustered_val_questions_vqa_format.json', 'w') as f:
    json.dump(val_data, f, indent=2)

print("处理完成!")