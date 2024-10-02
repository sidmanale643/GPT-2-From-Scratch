import torch
import torch.nn as nn
import tiktoken
from model import GPT2

config = {'d_model' : 768 ,'n_heads' : 8 , 'vocab_size' : 50257 , 'seq_len' : 1024 , 'dropout' : 0.1 , 'n_blocks' : 12}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT2(config).to(device)

model.load_state_dict(torch.load('gpt2_model_weights.pth', map_location=device))

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

out = generate_text(model , '')