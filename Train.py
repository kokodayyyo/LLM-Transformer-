import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

"""
这个脚本实现了一个简单的GPT语言模型，包括数据加载、模型定义、训练过程和模型保存。
"""

# 直接在代码中设置batch_size
batch_size = 48  # 或者任何您想要的批处理大小

print(f'batch size: {batch_size}')

# 检查是否有可用的CUDA设备，如果有则使用CUDA，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型的其他超参数
block_size = 50  # 每个序列的长度
max_iters = 20  # 增加训练迭代次数
learning_rate = 3e-4  # 学习率
eval_iters = 1  # 评估迭代次数
n_embd = 512  # 嵌入维度
n_head = 4  # 注意力头的数量
n_layer = 4  # Transformer块的层数
dropout = 0.2  # Dropout率
temperature = 0.8  # 温度参数，用于控制生成的随机性
max_grad_norm = 1.0  # 梯度裁剪的最大范数

print(device)

# 加载词汇表
vocab = {}
with open("vocab.txt", 'r', encoding='utf-8') as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)

vocab_size = len(vocab)


def parse_list_string(s):
    return [int(x.strip()) for x in s.strip('[]').split(',')]


class JokeDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_ids = torch.tensor(parse_list_string(row['Input']), dtype=torch.long)
        output_ids = torch.tensor(parse_list_string(row['Output']), dtype=torch.long)
        return input_ids, output_ids


# 创建训练和验证数据集
train_dataset = JokeDataset('train_data.csv')
val_dataset = JokeDataset('validation_data.csv')

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k, (xb, yb) in enumerate(loader):
            if k >= eval_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast('cuda'):
                logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * block_size - 1, head_size))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B,T,hs)

        # 计算相对位置编码
        rel_pos = torch.arange(T, device=x.device).unsqueeze(0) - torch.arange(T, device=x.device).unsqueeze(1)
        rel_pos = rel_pos.clamp(-block_size + 1, block_size - 1) + block_size - 1
        rel_pos_emb = self.rel_pos_emb[rel_pos]  # (T,T,hs)

        # 计算注意力权重
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # 确保rel_pos_emb的形状与q匹配
        rel_pos_emb = rel_pos_emb.unsqueeze(0).expand(B, -1, -1, -1)  # (B, T, T, hs)
        q_expanded = q.unsqueeze(2)  # (B, T, 1, hs)

        # 加入相对位置编码
        wei += torch.sum(q_expanded * rel_pos_emb, dim=-1)  # (B, T, T)

        # 动态创建tril矩阵
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, expansion_factor=4, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(n_embd, n_embd * expansion_factor))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(n_embd * expansion_factor, n_embd))
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # 这里我们需要调整损失计算，因为目标序列可能包含<PAD>
            mask = (targets != vocab['<PAD>']).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=vocab['<PAD>'])
            loss = (loss * mask.view(-1)).sum() / mask.sum()

        return logits, loss

    def generate(self, index, max_new_tokens, beam_size=5):
        # 初始序列
        sequences = [(index, 0.0)]  # (序列, 累积概率)

        for _ in range(max_new_tokens):
            all_candidates = []

            for seq, score in sequences:
                # 获取预测
                with autocast():
                    logits, _ = self.forward(seq)
                # 只关注最后一个时间步
                logits = logits[:, -1, :]  # 变为(B, C)
                # 应用softmax获取概率
                probs = F.softmax(logits / temperature, dim=-1)  # (B, C)

                # 获取前beam_size个最可能的下一个词
                top_probs, top_indices = probs.topk(beam_size)
                for i in range(beam_size):
                    next_seq = torch.cat((seq, top_indices[:, i].unsqueeze(1)), dim=1)
                    next_score = score - torch.log(top_probs[:, i])  # 负对数概率
                    all_candidates.append((next_seq, next_score.item()))

            # 排序所有候选序列，保留前beam_size个
            ordered = sorted(all_candidates, key=lambda x: x[1])
            sequences = ordered[:beam_size]

        # 返回最可能的序列
        return sequences[0][0]


# 创建模型实例
model = GPTLanguageModel(vocab_size).to(device)

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 学习率调度器 6次修改一次
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 混合精度训练的GradScaler
scaler = torch.amp.GradScaler('cuda')

# 训练循环
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"循环 {iter}: 训练损失 {losses['train']:.3f}, 验证损失 {losses['val']:.3f}")

    model.train()
    for batch in train_loader:
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast('cuda'):
            logits, loss = model(xb, yb)

        # 梯度裁剪
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # 更新学习率
    scheduler.step()

# 保存模型
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('模型已保存')
