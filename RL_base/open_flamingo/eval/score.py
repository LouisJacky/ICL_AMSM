import json
import logging

logging.basicConfig(level=logging.INFO)


def calculate_accuracy(data):
    correct = 0
    total = 0

    for item_index, item in enumerate(data):
        for prediction_index, prediction in enumerate(item):
            total += 1
            answers = prediction['answers']
            predicted_answer = prediction.get('predicted_answer')

            if predicted_answer is None:
                logging.warning(f"预测答案为None: 项目 {item_index}, 预测 {prediction_index}")
                continue

            predicted_answer = predicted_answer.lower()

            correct_count = sum(1 for answer in answers if answer.lower() in predicted_answer)

            if correct_count >= 3:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


# 读取JSON文件
try:
    with open('/path/to/Code/ICL_diversity_ofv2/RL_base/vizwiz_results_1percent_no_promt.json', 'r') as file:
        data = json.load(file)
except json.JSONDecodeError:
    logging.error("JSON文件格式错误")
    exit(1)
except FileNotFoundError:
    logging.error("找不到JSON文件")
    exit(1)

# 计算准确率
accuracy = calculate_accuracy(data)

print(f"预测准确率: {accuracy:.2%}")


