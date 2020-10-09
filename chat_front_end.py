# -*- coding: utf-8 -*-
import torch
import json
import random
from model import Net
from nltk_helper import stem, tokenize, bag_of_words

device = torch.device('cpu')
if torch.cuda.is_available() == True:
    device = torch.device('cuda')

data = torch.load('data.pth')

model_state = data['model_state']
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

model = Net(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    
chat_name = 'Pytorch-powered-BOT'
print(" Lets chat ! Type 'quit' to exit")

while True:
    sentence = input('You:')
    if sentence == 'quit':
        break
    
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)
    
    output = model(x)
    _ , pred = torch.max(output, dim=1)
    tag = tags[pred.item()]
    prob = torch.softmax(output, dim=1)
    actual_prob = prob[0][pred.item()]
    if actual_prob > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(chat_name, ':', random.choice(intent['responses']))
    else:
        print(chat_name,':', 'Sorry, I do not understand')