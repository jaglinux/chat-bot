# chat-bot
Pytorch based deep learning model for Chat-Bot application

1. pip install torch
2. pip install nltk

# you can skip step 3 and directly jump to step 4 which uses pre trained model present in data.pth
# Step 3 trains the model and saves the model into data.pth and set the save flag to False
# you can use saved mode data.pth and no need to train always.
3. python3 chat_train.py

4. python3 chat_front_end.py

Example:
You:hi
Pytorch-powered-BOT : Hi there, what can I do for you?

You:do you accept cash
Pytorch-powered-BOT : We accept most major credit cards, and Paypal

You:I need to go
Pytorch-powered-BOT : Have a nice day

