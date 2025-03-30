import torch
import pickle
import torch.nn as nn
from torch.nn import functional as F

# 定义模型的超参数
block_size = 50  # 每个序列的长度
n_embd = 512  # 嵌入维度
n_head = 2  # 注意力头的数量
n_layer = 2  # Transformer块的层数
dropout = 0.2  # Dropout率
temperature = 0.8  # 温度参数，用于控制生成的随机性

# 检查是否有可用的CUDA设备，如果有则使用CUDA，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载词汇表
vocab = {}
with open("vocab.txt", 'r', encoding='utf-8') as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)

vocab_size = len(vocab)


def parse_list_string(s):
    return [int(x.strip()) for x in s.strip('[]').split(',')]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # 动态创建tril矩阵
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,hs)
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
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

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

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            # 只关注最后一个时间步
            logits = logits[:, -1, :]  # 变为(B, C)
            # 应用softmax获取概率
            probs = F.softmax(logits / temperature, dim=-1)  # (B, C)
            # 从分布中采样
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将采样的索引添加到当前序列
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index


# 加载模型并将其移动到相应设备上
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
model = model.to(device)
model.eval()

# 设置生成文本的最大新token数量，可根据需要调整
max_new_tokens = 50

while True:
    # 获取用户输入的起始文本内容
    start_text = input("我会回应一些有趣的内容（输入 'exit' 可退出程序）：")
    if start_text.lower() == "exit":
        break
    # 将输入的起始文本转换为对应的索引序列
    input_index = []
    for word in start_text.split():
        if word in vocab:
            input_index.append(vocab[word])
        else:
            print(f"单词 '{word}' 不在词汇表中，将被忽略。")
    input_tensor = torch.tensor([input_index], dtype=torch.long).to(device)

    # 使用模型生成文本
    with torch.no_grad():
        output_index = model.generate(input_tensor, max_new_tokens)
        output_index = output_index.squeeze(0).tolist()

    # 将生成的索引转换回文本
    generated_words = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in output_index]
    generated_text = " ".join(generated_words)
    print("J：", generated_text)

print("程序已退出。")
