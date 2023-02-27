import os
import json
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
  BertTokenizerFast,
)
import config


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            try:
                self.preprocess(file_name)
            except:
                continue

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        f=open(input_dir, 'r', encoding='utf-8')
        lines = f.readlines()
        # 先读取到内存中，然后逐行处理
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        for line in tqdm(lines):
            line = line.replace('\\xa0','')
            # loads()：用于处理内存中的json对象，strip去除可能存在的空格
#             print(line)
            try:
                json_line = json.loads(line.strip())
    #             print(json_line['text'])
                words = []
                text = json_line['text']
                bad_tokens = [' ','\n',' ',' ','​','  ',' �','   ',' 　 ',' � ',' ﻿ ',' 	 ',' 	 ',' � ','   ',' ﻿ ',' 　 ',' 　 ',
                              ' 　 ',' 　 ',' 　 ',' ‍ ',' 　 ','   ','   ',' 　 ',' ﻿ ','   ']
                for token in text:
                    token = token.lower()
    #                 print(token)
                    if token in bad_tokens:
                        words.append('.')
                        continue
                    token = token.replace("“",'"').replace("”",'"')
                    if tokenizer.tokenize(token):
                        words.append(token)
                    else:
                        print('bad token：【',token,'】')
                        bad_tokens.append(token)
                        words.append('.')          

    #             words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)
                flag = 0
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
    #                             print(sub_name, ''.join(words[start_index:end_index + 1]))
                                if ''.join(words[start_index:end_index + 1]) == sub_name:
                                    flag = 1
                                    if start_index == end_index:
                                        labels[start_index] = 'S-' + key
                                    else:
                                        labels[start_index] = 'B-' + key
                                        labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                if len(words) < 510 and flag == 1:
    #                 print(words,labels)
                    word_list.append(words)
                    label_list.append(labels)
            except:
                continue
            # 保存成二进制文件
#         print(words,labels)
        np.savez_compressed(output_dir, words=word_list, labels=label_list)
        logging.info("--------{} data process DONE!--------".format(mode))



# 对于干净数据可以使用以下代码，不然可能报错：mask of the first timestep must all be on
# class Processor:
#     def __init__(self, config):
#         self.data_dir = config.data_dir
#         self.config = config

#     def process(self):
#         """
#         process train and test data
#         """
#         for file_name in self.config.files:
#             try:
#                 self.preprocess(file_name)
#             except:
#                 continue

#     def preprocess(self, mode):
#         """
#         params:
#             words：将json文件每一行中的文本分离出来，存储为words列表
#             labels：标记文本对应的标签，存储为labels
#         examples:
#             words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
#             labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
#         """
#         input_dir = self.data_dir + str(mode) + '.json'
#         output_dir = self.data_dir + str(mode) + '.npz'
#         if os.path.exists(output_dir) is True:
#             return
#         word_list = []
#         label_list = []
#         f=open(input_dir, 'r', encoding='utf-8')
#         lines = f.readlines()
#         # 先读取到内存中，然后逐行处理
# #         tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
#         for line in tqdm(lines):
# #             line = line.replace('\\xa0','')
#             # loads()：用于处理内存中的json对象，strip去除可能存在的空格
# #             print(line)
#             try:
#                 json_line = json.loads(line.strip())
#                 # print(json_line['text'])
#                 words = []
#                 text = json_line['text']

#                 for token in text:
#                     token = token.lower()
#                     token = token.replace("“",'"').replace("”",'"')
#                     words.append(token)
        
#     #             words = list(text)
#                 # 如果没有label，则返回None
#                 label_entities = json_line.get('label', None)
#                 labels = ['O'] * len(words)

#                 flag = 0
#                 if label_entities is not None:
#                     for key, value in label_entities.items():
#                         for sub_name, sub_index in value.items():
#                             for start_index, end_index in sub_index:
#     #                             print(sub_name, ''.join(words[start_index:end_index + 1]))
#                                 if ''.join(words[start_index:end_index + 1]) == sub_name:
#                                     flag = 1
#                                     if start_index == end_index:
#                                         labels[start_index] = 'S-' + key
#                                     else:
#                                         labels[start_index] = 'B-' + key
#                                         labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
#                 if len(words) < 510 and flag == 1:
#     #                 print(words,labels)
#                     word_list.append(words)
#                     label_list.append(labels)
#             except:
#                 continue
#             # 保存成二进制文件
# #         print(words,labels)
#         np.savez_compressed(output_dir, words=word_list, labels=label_list)
#         logging.info("--------{} data process DONE!--------".format(mode))