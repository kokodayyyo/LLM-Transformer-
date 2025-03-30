import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
import contractions

# 下载NLTK的分词器
nltk.download('punkt')


def clean_text(text):
    # 移除编号和引号
    text = re.sub(r'^\d+,"', '', text)  # 移除开头的编号和引号
    text = re.sub(r'"$', '', text)  # 移除结尾的引号
    # 处理特殊字符和表情符号
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # 移除非ASCII字符（包括表情符号）
    # 处理缩写
    text = contractions.fix(text)
    # 转换为小写
    text = text.lower()
    # 处理多余的空格和换行符
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_joke(joke, max_length=30):
    # 使用NLTK进行分词
    tokens = word_tokenize(joke)
    # 截断或填充
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += ['<PAD>'] * (max_length - len(tokens))
    return tokens


def build_vocab(tokens_list, vocab_size=50000):
    # 统计所有单词的频率
    all_tokens = [token for tokens in tokens_list for token in tokens]
    word_freq = Counter(all_tokens)

    # 构建词汇表
    vocab = {word: idx for idx, (word, _) in enumerate(word_freq.most_common(vocab_size), start=1)}
    vocab['<UNK>'] = 0  # 未知词标记
    vocab['<PAD>'] = len(vocab)  # 填充标记
    vocab['<SOS>'] = len(vocab)  # 开始标记
    vocab['<EOS>'] = len(vocab)  # 结束标记

    return vocab


def encode_joke(tokens, vocab):
    # 将单词转换为索引
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]


def create_input_output_pair(tokens, vocab, max_length=50):
    # 创建输入-输出对
    input_tokens = tokens[:max_length - 1] + ['<SOS>']
    output_tokens = tokens[1:] + ['<EOS>']
    return encode_joke(input_tokens, vocab), encode_joke(output_tokens, vocab)


def preprocess_data(file_path, vocab_size=50000, max_length=50):
    # 读取CSV文件
    df = pd.read_csv(file_path, encoding='utf-8')

    # 检查空值
    df = df.dropna(subset=['Joke'])

    # 去除重复的笑话
    df = df.drop_duplicates(subset=['Joke'])

    # 清洗文本
    df['Joke'] = df['Joke'].apply(clean_text)

    # 分词处理
    df['Tokens'] = df['Joke'].apply(lambda x: tokenize_joke(x, max_length))

    # 构建词汇表
    vocab = build_vocab(df['Tokens'].tolist(), vocab_size)

    # 编码笑话并创建输入-输出对
    df[['Input', 'Output']] = df['Tokens'].apply(
        lambda tokens: pd.Series(create_input_output_pair(tokens, vocab, max_length)))

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)  # 10% of 80% = 8% of total

    return train_df, val_df, test_df, vocab


if __name__ == "__main__":
    file_path = "shortjokes.csv"
    train_df, val_df, test_df, vocab = preprocess_data(file_path)

    # 保存处理后的数据集和词汇表
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('validation_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    # 保存词汇表
    with open('vocab.txt', 'w') as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\n")

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"词汇表大小: {len(vocab)}")
