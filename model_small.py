from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
from transformers import BertTokenizer, BertModel
import config
# device = config.device
# b = torch.eye(22)
# b[2][2] = 0.5
# #             b[9][9] = 0.5
# #             b[16][16] = 0.5
# #             print(device)
# b = b.to(device)

class BertNER_mengzi(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER_mengzi, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained("Langboat/mengzi-bert-base")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # lstm_embedding_size=128,
        # lstm_dropout_prob=0.5
        # self.bilstm = nn.LSTM(
        #     input_size=lstm_embedding_size,  # 1024
        #     hidden_size=config.hidden_size // 2,  # 1024
        #     batch_first=True,
        #     num_layers=2,
        #     dropout=lstm_dropout_prob,  # 0.5
        #     bidirectional=True
        # )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(padded_sequence_output)
        # logits = padded_sequence_output
        outputs = (logits,)
        if labels is not None:#如果标签存在就计算loss，否则就是输出线性层对应的结果，这样便于通过后续crf的decode函数解码得到预测结果。
            loss_mask = labels.gt(-1)
#             print(logits.size(), labels, loss_mask)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
    
class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_pretrained('ckiplab/albert-tiny-chinese')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # lstm_embedding_size=128,
        # lstm_dropout_prob=0.5
        # self.bilstm = nn.LSTM(
        #     input_size=lstm_embedding_size,  # 1024
        #     hidden_size=config.hidden_size // 2,  # 1024
        #     batch_first=True,
        #     num_layers=2,
        #     dropout=lstm_dropout_prob,  # 0.5
        #     bidirectional=True
        # )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(padded_sequence_output)
        # logits = padded_sequence_output
        outputs = (logits,)
        if labels is not None:#如果标签存在就计算loss，否则就是输出线性层对应的结果，这样便于通过后续crf的decode函数解码得到预测结果。
            loss_mask = labels.gt(-1)
            
            # print(logits.size(), labels.size(), loss_mask.size())
            # for i in range(32):
                # print(labels[i])
                # print(loss_mask[i])
            # loss = self.crf(torch.matmul(logits,b), labels, loss_mask) * (-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
