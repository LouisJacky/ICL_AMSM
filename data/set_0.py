import json

# 读取JSON文件
json_file_path = '/path/to/Code/Painter/Painter/toy_datasets/derain/derain_train.json'
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 修改每个item的'type'字段
for item in data:
    item['type'] = '0'

# 将修改后的数据写回JSON文件
with open("/path/to/Code/Painter/Painter/toy_datasets/derain/derain_train_clustered.json", 'w') as f:
    json.dump(data, f, indent=4)

print("JSON文件已更新。")