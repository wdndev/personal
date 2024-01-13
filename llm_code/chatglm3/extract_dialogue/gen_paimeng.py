import json
from tqdm import tqdm
import pandas as pd

def read_txt(file_path):
    # path:字符串形式
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def read_excel_column(excel_file, sheet_name="paimeng", column_index=1):
    try:
        # 读取Excel文件指定sheet的内容
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 获取指定列的数据
        column_data = df.iloc[:, column_index].tolist()
        
        return column_data
    except Exception as e:
        print(f"Error: {e}")
        return []

def save_dataset(path, data):
    # filename = path.split('/')[-1].split('.')[0]
    with open(path, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

def gen_dataset(dialogue_list):
    res = []
    print('构造微调数据集')
    for i in tqdm(range(1, len(dialogue_list))):
        tmp = {
            "instruction": dialogue_list[i-1],
            "input": "",
            "output": dialogue_list[i]
        }
        res.append(tmp)
    return res

if __name__ == "__main__":
    dialogue_list = read_excel_column(excel_file="input\genshin_impact_4.0.xlsx")
    print(f"共有 {len(dialogue_list)} 条对话样本")
    dataset = gen_dataset(dialogue_list)
    print(f"获得 {len(dataset)} 条微调样本")
    save_dataset("output/paimeng.json", dataset)