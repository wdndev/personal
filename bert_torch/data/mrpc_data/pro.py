# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import os
import tokenization

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def get_train_examples(input_file, result_path):
    """See base class."""
    lines = _read_tsv(input_file)
    txt_str = ""
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = line[3]
      text_b = line[4]
      txt_str += text_a + " \t " + text_b + " \n"
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(txt_str)

def process_train(mrpc_train_path, result_path):
    data = pd.read_csv(mrpc_train_path, sep='\t', header=0,error_bad_lines=False)
    filtered_data = data[data['Quality'] == 1]

    txt_str = ""
    # 遍历每一行
    for index, row in filtered_data.iterrows():
        # 将第二列和第三列转换为字符串
        column2_str = str(row['#1 String'])
        column3_str = str(row['#2 String'])
        if column2_str == '' or column3_str=="":
            continue
        txt_str += column2_str
        txt_str += " \\t "
        txt_str += column3_str
        txt_str += "\\n \n"

        # print(index)
        # print(column2_str)
        # print(column3_str)

        # if index == 5:
        #     break
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(txt_str)

def process_test(mrpc_test_path, result_path):
    data = pd.read_csv(mrpc_test_path, sep='\t', header=0,error_bad_lines=False)

    txt_str = ""
    # 遍历每一行
    for index, row in data.iterrows():
        # 将第二列和第三列转换为字符串
        column2_str = str(row['#1 String'])
        column3_str = str(row['#2 String'])
        if column2_str == '' or column3_str=="":
            continue

        txt_str += column2_str
        txt_str += ' \\t '
        txt_str += column3_str
        txt_str += '\\n \n'

        # print(index)
        # print(column2_str)
        # print(column3_str)

        # if index == 5:
        #     break
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(txt_str)




if __name__ == '__main__' :
    get_train_examples("train.tsv", "train.txt")
    get_train_examples("test.tsv", "test.txt")