import pandas as pd

import torch

from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import nn

from classifier import Classifier
from dataset import Dataset

df = pd.read_table('../data/train.tsv.zip')
df.head()

def train(model, data, learning_rate, epochs):
    train_data, val_data = train_test_split(data, test_size = 0.1)
    print("train data size: {}, test data size: {}\n".format(len(train_data), len(val_data)))

    train, val = Dataset(train_data), Dataset(val_data)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    mean = 0
    for epoch_num in range(epochs):
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        acc_val = total_acc_val / len(val_data)
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} '\
            f'| Train Accuracy: {total_acc_train / len(train_data): .3f} '\
            f'| Val Loss: {total_loss_val / len(val_data): .3f} '\
            f'| Val Accuracy: {acc_val: .3f}')
        mean += acc_val
    
    mean /= 3
    print('\nAverage accuracy of 3 times: {mean: .3f}')
                  
def main():
    EPOCHS = 3
    model = Classifier()
    LR = 1e-5
    train(model, df, LR, EPOCHS)

    # save the model we have trained
    torch.save(model.state_dict(), "bert.model")

if __name__ == '__main__':
    main()