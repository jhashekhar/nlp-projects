import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


class BertDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    # i want to get out a vector of numbers that /
    # can be used as input for the transformer model
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = str(self.texts[idx])
        target = self.targets[idx]
        #tokenizer = BertTokenizer.from_pretrained('vocab.txt')
        inputs = self.tokenizer.encode(text, 
                                       add_special_tokens=True, 
                                       max_length=self.max_len)
        
        # padding - forgot to do the padding threw a size of tensor mismatch error
        padding_len = self.max_len - len(inputs)
        inputs = inputs + ([0] * padding_len)
        # print(torch.tensor(inputs, dtype=torch.long).shape)
        return {'input_ids': torch.tensor(inputs, dtype=torch.long), 
                'targets': torch.tensor(target, dtype=torch.float)}


class BERTModel(nn.Module):
    def __init__(self, model_path):
        super(BERTModel, self).__init__()
        self.path = model_path
        self.model = BertModel.from_pretrained(self.path)
        self.fc = nn.Linear(768, 64)
        self.fc1 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        #self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        _, out = self.model(inputs)
        out = self.dropout(self.relu(self.fc(out)))
        out = self.dropout(self.fc1(out))
        return self.sigmoid(out)


def loss_fn(outputs, targets):
    return nn.BCELoss()(outputs, targets)


def train_loop(data_loader, model,
                  optimizer, device, scheduler=None):
    model.train()
    # Components:-
    # load data from dataloader
    for bi, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        targets = data['targets'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        #print(outputs, np.round(outputs.cpu().detach().numpy()))
        #print(outputs.shape, targets.unsqueeze(1).shape)
        loss = loss_fn(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if bi % 170 == 0:
            print(f"bi: {int(bi/10)} \t loss: {loss:.4f}")
        


def eval_loop(data_loader, model, device):
    
    model.eval()
    final_targets = []
    final_outputs = []

    for bi, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        targets = data['targets'].to(device)
        outputs = model(input_ids)
        val_loss = loss_fn(outputs, targets)

        final_outputs.append(outputs.cpu().detach().numpy())
        final_targets.append(targets.unsqueeze(1).cpu().detach().numpy())

    # np.vstack is to stack the sequence of input arrays       
    return np.vstack(final_outputs), np.vstack(final_targets), val_loss


def run():

    device = 'cuda'
    BATCH_SIZE = 16
    EPOCHS = 10
    lr = 3e-5
    MAX_LEN = 160
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#.save_pretrained('.')
    model = BERTModel('bert-base-uncased').to(device)
    

    optimizer = AdamW(model.parameters(), lr=lr)


    train_df = pd.read_csv('train.csv')
    X_train, X_valid = train_test_split(train_df, test_size=0.1, random_state=42)
    # Datasets
    train_dataset = BertDataset(
    texts = X_train.text.values,
    targets = X_train.target.values,
    tokenizer = tokenizer, 
    max_len = MAX_LEN)

    valid_dataset = BertDataset(
        texts = X_valid.text.values, 
        targets = X_valid.target.values,
        tokenizer = tokenizer,
        max_len = MAX_LEN)


    # DataLoaders
    train_loader = DataLoader(
        train_dataset,                      
        batch_size=BATCH_SIZE, 
        shuffle=True)

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True)


    for epoch in range(EPOCHS):
        print(f"Training...")
        train_loop(train_loader, model, optimizer,'cuda')
        
        o, t, val_loss = eval_loop(data_loader=valid_loader, model=model, device='cuda')
        acc = accuracy_score(t, np.round(o), normalize=True)
        print(f"Epoch: {epoch + 1}/{EPOCHS} \t accuracy: {acc} \t val loss: {val_loss}")
        

run()
    
