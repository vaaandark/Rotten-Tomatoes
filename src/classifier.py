from torch import nn
from transformers import BertModel

# 使用 Bert 模型分类
class Classifier(nn.Module):

    def __init__(self, dropout=0.1):
        super(Classifier, self).__init__()

        # 使用 bert-tiny 预处理模型
        # https://huggingface.co/prajjwal1/bert-tiny
        # 它是一个 Pytorch 预训练的模型，由官方 Google BERT 存储库中的 Tensorflow 检查点转换而得
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')

        # 神经网络
        # 忽略单元量默认为 0.1
        self.dropout = nn.Dropout(dropout)

        # 线性变换
        # 最后分成五个等级，与烂番茄评分对应
        self.linear = nn.Linear(128, 5)

        # Rectified Linear Units - 修正线性单元
        # ReLU(x)==max(0,x)
        self.relu = nn.ReLU()

    # 前向传播
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # 返回最后一层
        final_layer = self.relu(linear_output)
        return final_layer 