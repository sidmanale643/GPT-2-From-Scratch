from altair import sample
import torch
import torch.nn as nn
import math
import tiktoken

class Norm(nn.Module):
    def __init__(self , d_model):
        super().__init__()      
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self , x):
        return self.norm(x)

class Embeddings(nn.Module):
    def __init__(self , vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size , d_model)
        
    def forward(self , x):
        embeddings = self.embedding(x)
        return embeddings
       
class MHA(nn.Module):
    def __init__(self , d_model , n_heads ):
        super().__init__()
        
        assert d_model % n_heads  == 0
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model , d_model , bias = False)
        self.w_k = nn.Linear(d_model , d_model , bias = False)
        self.w_v = nn.Linear(d_model , d_model , bias = False)
        self.w_o = nn.Linear(d_model , d_model , bias= False)
        
    def forward(self , Q , K , V ):
        b , n , d_model = Q.size()
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        Q = Q.view(b , n  , self.n_heads , self.d_k).transpose(1,2)
        K = K.view(b , n  , self.n_heads , self.d_k).transpose(1,2)
        V = V.view(b , n  , self.n_heads , self.d_k).transpose(1,2)
        
        attention_scores = torch.matmul(Q , K.transpose(-2 , -1)) / math.sqrt(self.d_k)
        
        mask = torch.tril(torch.ones(n , n).bool()).unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(mask == 0 , -1e9)
        attention_weights = torch.softmax(attention_scores , dim = -1)
        attention_out =  torch.matmul(attention_weights , V)
        out = attention_out.transpose(1 , 2).contiguous().view(b ,n , d_model)    
        out = self.w_o(out)
        return out
       
class FFN(nn.Module):
    def __init__(self , d_model , dropout ):
        super().__init__()

        self.fc1 = nn.Linear(d_model , 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model , d_model)
        self.gelu = nn.GELU(approximate= 'tanh')
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))
    
class DecoderBlock(nn.Module):
    def __init__(self , d_model , n_heads , dropout):
        super().__init__()
        
        self.mha = MHA(d_model , n_heads)
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.ffn = FFN(d_model , dropout)

    def forward(self , x ):
        x = self.norm1(x)
        x = x + self.mha(x , x , x)
        x = self.norm2(x)
        x = self.ffn(x)
        return x 
    
class Decoder(nn.Module):
    def __init__(self , d_model , n_heads , seq_len , vocab_size , dropout , n_blocks):
        super().__init__()
        
        self.pos = nn.Embedding(seq_len , d_model)
        self.embeddings = Embeddings(vocab_size , d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model , n_heads , dropout) for _ in range(n_blocks)])
        
    def forward(self , x):
        
        embeddings = self.embeddings(x)
        position_ids = torch.arange(0, x.size(1)).unsqueeze(0).expand(x.size(0), -1)
        pos_embeddings = self.pos(position_ids)
        decoder_in = embeddings + pos_embeddings
        for layer in self.layers:
            decoder_in = layer(decoder_in)
        return decoder_in

config = {'d_model' : 768 ,'n_heads' : 8 , 'vocab_size' : 50257 , 'seq_len' : 1024 , 'dropout' : 0.1 , 'n_blocks' : 12}

class GPT2(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.decoder =  Decoder(config['d_model'], config['n_heads'], config['seq_len'], config['vocab_size'], config['dropout'], config['n_blocks'])
        self.fc = nn.Linear(config['d_model'] , config['vocab_size'] )
    
    def forward(self , x):
        logits = self.fc(self.decoder(x))
        return logits 
    
model = GPT2(config)
print(model)

#Sample Inputs
batch_size = 3
input = torch.randint(0, config['vocab_size'], (batch_size, config['seq_len']))

