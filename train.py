import tiktoken
import torch
import torch.nn as nn
import os
from gpt2 import GPT2

file_path = 'E:\ml\decoder-only\pg2600.txt'

with open(file_path, 'r') as file:
    text = file.read()

enc = tiktoken.encoding_for_model("gpt2")

    
config = {'d_model' : 768 ,'n_heads' : 8 , 'vocab_size' : 50257 , 'seq_len' : 1024 , 'dropout' : 0.1 , 'n_blocks' : 12}
model = GPT2(config)
text = text[:1000]
tokens =  enc.encode(text)
B, T = 4 , 32
buf = torch.tensor(tokens[:B*T + 1])
x = buf[:-1].view(B , T)
y = buf[1:].view(B , T)
print(x.shape)
logits , loss = model(x , y)
print(loss)

