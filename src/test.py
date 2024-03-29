#!env python3
# 根据之前保存的模型判断你的评论对应的评分
# 用法：
#   python3 test.py [{你的评论（英文）}]
from classifier import Classifier
import torch
import sys
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

model = Classifier()
model.load_state_dict(torch.load("bert.model"))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    model = model.cuda()

test_text = "I think this movie is great"
if len(sys.argv) > 1:
  test_text = sys.argv[1]

with torch.no_grad():
    test_input = tokenizer(test_text, padding='max_length',
        max_length=512, truncation=True, return_tensors="pt")
    mask = test_input['attention_mask'].to(device)
    input_id = test_input['input_ids'].squeeze(1).to(device)

    output = model(input_id, mask)
    label = output.argmax(dim=1).item()
    print("Label:", label)