# coding=UTF-8
import os
import torch
data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
predict_dir = data_dir + 'predict.npz'
files = ['train', 'test', 'predict']
model_dir = os.getcwd() + '/experiments/s_model/'
log_dir = model_dir + 'train.log'

# bert_model = '/data/fuwen/SuWen/Bert-classification/chinese-bert-wwm'
# roberta_model = '/data/fuwen/SuWen/Bert-classification/chinese-bert-wwm'
# model_dir = os.getcwd() + '/experiments/s_model/'

# case_dir = os.getcwd() + '/case/bad_case.txt'
# predict_seq_output_dir = '/data/fuwen/SuWen/news_get_name_ner/src/predict/s_seq_output.txt'
# predict_dict_output_dir = '/data/fuwen/SuWen/news_get_name_ner/src/predict/s_dict_output.txt'
# predict_entity_list_dir = '/data/fuwen/SuWen/news_get_name_ner/src/predict/s_entity_list_output.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 128
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

device = torch.device("cuda:0")
# device = torch.device("cpu")
print(device)
labels = ['element'] # 如果不区分抽什么，可以用这个，适用于老版本数据
# 形如：{"text": "连续生产劲头足 福建华兴玻璃力促首季产值提升", "label": {"element": {"华兴玻璃": [[10, 13]]}}}
label2id = {
    "O": 0,
    "B-element": 1,
    "I-element": 2,
    "S-element": 3,
}
num_labels = 4

# labels = ['event','company','chanye','loc','person']
# # 数据形如：{"text": "黄金交易提醒：美国经济数据惨淡，特朗普意外拒签纾困案，为后市金价提供更多悬念", "label": {"loc": {"美国": [[7, 8]]}, "person": {"特朗普": [[16, 18]]}}}
# label2id = {
#     "O": 0,
#     "B-event": 1,
#     "B-company": 2,
#     "B-chanye": 3,
#     "B-loc": 4,
#     "B-person": 5,
#     "I-event": 6,
#     "I-company": 7,
#     "I-chanye": 8,
#     "I-loc": 9,
#     "I-person": 10,
#     "S-event": 11,
#     "S-company": 12,
#     "S-chanye": 13,
#     "S-loc": 14,
#     "S-person": 15,
# }

# num_labels = 16



hidden_dropout_prob = 0.1
id2label = {_id: _label for _label, _id in list(label2id.items())}
