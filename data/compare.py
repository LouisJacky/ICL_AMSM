import torch
import numpy as np

def compare_pth_files(file1_path, file2_path):
    # 加载两个.pth文件
    state_dict1 = torch.load(file1_path, map_location=torch.device('cpu'))
    state_dict2 = torch.load(file2_path, map_location=torch.device('cpu'))

    # 检查两个文件是否包含相同的键
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        print("两个文件包含的参数键不完全相同。")
        return False

    # 比较每个参数的值
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"参数 '{key}' 在两个文件中的值不同。")
            return False

    print("两个.pth文件中的所有参数值完全相同。")
    return True

# 使用示例
file1_path = "../log/ofv2_base/END_OUTPUT_linear/vizwiz/vizwiz_projector_0.pt"
file2_path = "/data16tb/ljq/checkpoints/ofv2/checkpoint.pt"

are_identical = compare_pth_files(file1_path, file2_path)