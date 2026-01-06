import torch
import torch.nn as nn
import time
import pandas as pd
import sys

# Configuration for 110M TinyLlama
# Based on stories110M.bin architecture
CONFIG = {
    'dim': 768,
    'n_layers': 12,
    'n_heads': 12,
    'vocab_size': 32000,
    'max_seq_len': 1024
}

class TinyLlamaDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(CONFIG['vocab_size'], CONFIG['dim'])
        self.layers = nn.ModuleList()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['dim'], 
            nhead=CONFIG['n_heads'], 
            dim_feedforward=int(4 * CONFIG['dim'] * 2/3), # SwiGLU approx
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG['n_layers'])
        self.norm = nn.LayerNorm(CONFIG['dim'])
        self.output = nn.Linear(CONFIG['dim'], CONFIG['vocab_size'], bias=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        x = self.encoder(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

def main():
    print(f"[*] Initializing PyTorch Benchmark (110M Params)...")
    print(f"    Config: {CONFIG}")
    
    device = torch.device('cpu')
    torch.set_num_threads(1) # Fair comparison: Single Thread
    
    model = TinyLlamaDummy().to(device)
    model.eval()
    
    # Warmup
    input_ids = torch.randint(0, CONFIG['vocab_size'], (1, 32)).to(device)
    print("[*] Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)

    print("[*] Running Benchmark (N=256 steps)...")
    latencies = []
    
    # The C++ bench does: forward(token, pos). It processes 1 token.
    # So we should benchmark 1->1 forward pass.
    
    input_tensor = torch.zeros((1, 1), dtype=torch.long).to(device)
    
    with torch.no_grad():
        for i in range(256):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1e6) # microseconds

    # Save Results
    df = pd.DataFrame(latencies, columns=['latency_us'])
    df.to_csv('pytorch_results.csv', index=False)
    
    avg_lat = df['latency_us'].mean()
    print(f"[*] Done. Average Latency: {avg_lat:.2f} us")
    print(f"[*] Throughput: {1e6/avg_lat:.2f} tok/s")

if __name__ == "__main__":
    main()
