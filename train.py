import time
import torch
import torch.nn as nn
from model import GPT2 , config
import tiktoken
import torch._dynamo
from torch.optim.lr_scheduler import OneCycleLR
torch._dynamo.config.suppress_errors = True
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision('high')
class Dataloader:
    def __init__(self , B , T):
        self.B = B
        self.T = T
        
        with open('E:/ml/decoder-only/wiki.train.txt', 'r') as f:
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
model =torch.compile(model)

def generate_text(model, start_text, max_length=200, temperature=0.8):
    enc = tiktoken.get_encoding('gpt2')
    context = torch.tensor(enc.encode(start_text)).unsqueeze(0).to(device)
    
    generated = context
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(generated[:, -config['seq_len']:])
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            
            if next_token.item() == enc.encode('\n')[0]:
                break
    
    return enc.decode(generated[0].tolist())

train_loader = Dataloader(4, 1024)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 , weight_decay = 0.01)
num_epochs = 10
steps_per_epoch = len(train_loader.tokens) // (train_loader.B * train_loader.T)
total_steps = num_epochs * steps_per_epoch

scheduler = OneCycleLR(
    optimizer,
    max_lr= 5e-4,
    total_steps=total_steps,
    pct_start=0.05,  
    anneal_strategy='cos',
    cycle_momentum=False
)


start_texts = [
    "I am a Language Model ",
    "Climate change impacts biodiversity by...",
    "In the 20th century, industrialization led to...",
    "Photosynthesis allows plants to convert sunlight into...",
    "As of 2024, the global economy is shaped by...",
    "What is the meaning of life? Philosophers argue that...",
    "Social media has transformed communication, leading to...",
    "AI advancements are revolutionizing healthcare by...",
]

for epoch in range(num_epochs):
    model.train()
    for i in range(steps_per_epoch):
        t1 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device) 
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss = loss.to(device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        t2 = time.time()
        diff = (t2 - t1) * 1000
        toks_ps = (train_loader.B * train_loader.T) / (t2 - t1)
        print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{steps_per_epoch}, Loss {loss.item():.4f}, Time {diff:.2f} ms, Tokens per sec {toks_ps:.2f}")
    
    if epoch % 2 == 0:
        print(f"\n--- Generated Sequences after Epoch {epoch+1} ---")
        for j, start_text in enumerate(start_texts, 1):
            generated = generate_text(model, start_text, max_length= 200)
            print(f"{j}. {generated}\n")
        print("--- End of Generated Sequences ---\n")

