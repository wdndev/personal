# -*- coding: utf-8 -*-
#  @file        - dataset.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 数据集类
#  @version     - 0.0
#  @date        - 2024.03.03
#  @copyright   - Copyright (c) 2024

class Dataset(object):
    def __init__(self):
        self.datasets = []

    def build(self, df, fea_col:str, label_col:str):
        for i in range(len(df)):
            data = df.iloc[i]
            text = data[fea_col]
            label = data[label_col]
            item = {
                "text": text,
                "label": label
            }
            self.datasets.append(item)

    def __iter__(self):
        for data in self.datasets:
            yield data
        
    def __getitem__(self, index):
        return self.datasets[index]
    
    def __len__(self):
        return len(self.datasets)


