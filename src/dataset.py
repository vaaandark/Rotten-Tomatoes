import numpy as np
import torch
from transformers import BertTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [label for label in df['Sentiment']]
        # 使用 bert-tiny 预处理模型
        # https://huggingface.co/prajjwal1/bert-tiny
        # 用于分词，将纯文本转换为编码，并添加标记，然后将这些词转换为字典索引
        tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt") for text in df['Phrase']]
    
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])
        return batch_texts, batch_y