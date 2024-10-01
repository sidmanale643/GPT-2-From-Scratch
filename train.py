import time
import torch
import torch.nn as nn
from gpt2 import GPT2 , config
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision('high')
class Dataloader:
    def __init__(self , B , T):
        self.B = B
        self.T = T
        
        with open('E:/ml/decoder-only/lyrics.txt', 'r') as f:
            text = f.read()   
        enc = tiktoken.get_encoding('gpt2')   
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 Epoch = { len(self.tokens) // (self.B * self.T)} batches")
        
        self.current_pos = 0
    
    def next_batch(self):
        B , T = self.B , self.T
        buffer = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buffer[:-1].view(B , T)
        y = buffer[1:].view(B , T)
        
        self.current_pos += B * T 
        
        if self.current_pos + (B * T + 1) > len(self.tokens):
            self.current_pos = 0

        return x , y

model = GPT2(config).to(device)
model = torch.compile(model)
train_loader= Dataloader(4, 32)
optimizer = torch.optim.AdamW(model.parameters() , lr = 3e-4)
loss_fn = nn.CrossEntropyLoss()

for i in range(100):
    
    t1 = time.time()
    x , y = train_loader.next_batch()
    x , y = x.to(device) , y.to(device) 
    optimizer.zero_grad()
    logits , loss  = model(x , y)
    loss = loss.to(device)
    loss.backward()
    optimizer.step()
    t2 = time.time()
    diff = (t2 - t1)*1000
    toks_ps = (train_loader.B * train_loader.T) / (t2 - t1)
    print(f"Step {i} Loss {loss.item()} Time {diff} ms  Tokens per sec {toks_ps}" )

def inference(input , vocab):
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(input))
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits , dim = -1)
        print(vocab[torch.argmax(probs)])
        
