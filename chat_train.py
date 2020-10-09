# -*- coding: utf-8 -*-
import torch
import json

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from model import Net
from nltk_helper import stem, tokenize, bag_of_words

with open('intents.json') as f:
    intents = json.load(f)

tags=[]
all_words=[]
input_vector=[]
unwanted = ['?', '.', '!']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        new_words=[]
        words = tokenize(pattern)
        for word in words:
            if word not in unwanted:
                new_words.append(word)
        new_words = [stem(word) for word in new_words]
        all_words.extend(new_words)
        input_vector.append((new_words, tag))

all_words = list(set(all_words))

all_words.sort()
#print(all_words)
tags = list(set(tags))
tags.sort()

x_train, y_train = [] , []

for (a,b) in input_vector:
    bag = bag_of_words(a, all_words)
    x_train.append(bag)
    y_train.append(tags.index(b))

x_train = np.array(x_train)
y_train = np.array(y_train)

#print(x_train)
#print(y_train)

#hyper parameters
epochs = 1000*5
batch = 8
lrr = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

#print(input_size, output_size)

class Chat(Dataset):
    def __init__(self):
        self.len = len(x_train)
        self.x_train = x_train
        self.y_train = y_train
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        return self.x_train[i], self.y_train[i]
    
dataset = Chat()
#print(len(dataset))

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    
train_loader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, num_workers=0)
model = Net(input_size, hidden_size, output_size)
# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lrr)


for epoch in range(epochs):
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(dtype=torch.long).to(device)
        
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch :', epoch, 'loss is ', loss.item())
        
save_data = {
'model_state' : model.state_dict(),
'input_size' : input_size,
'hidden_size' : hidden_size,
'output_size' : output_size,
'all_words' : all_words,
'tags' : tags
}
save = True
if save == True:        
    torch.save(save_data, 'data.pth')

print(save_data)
        

        
        

