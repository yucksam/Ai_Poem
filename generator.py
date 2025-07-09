import os
import json
import torch
import re
from collections import Counter
from typing import Dict

# ========== 模型缓存区域 ==========
_cached = {}  # 全局缓存：词表 + 模型


# ========== 词表加载 ==========
def load_vocab(file_path=r"model\tokenized_poems.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"词表文件 {file_path} 不存在")

    with open(file_path, "r", encoding="utf-8") as f:
        poems = json.load(f)

    all_tokens = [token for poem in poems for para in poem['paragraphs'] for token in para]
    cnt = Counter(all_tokens)
    tokens = ['<PAD>', '<START>', '<END>', '<UNK>'] + [t for t, c in cnt.items() if c >= 5]
    token2idx = {t: i for i, t in enumerate(tokens)}
    idx2token = {i: t for t, i in token2idx.items()}
    vocab_size = len(tokens)
    return token2idx, idx2token, vocab_size


# ========== 模型加载 ==========
def load_model_and_generator(model_type, strategy, token2idx, idx2token, vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == "rnn":
        from rnnmodel import PoemRNN, EnhancedPoemGenerator
        model = PoemRNN(vocab_size)
        model_path = r"D:/Desktop/AiPoem/model/poem_rnn.pth"
        generator = EnhancedPoemGenerator(token2idx, idx2token, str(device))

    elif model_type == "transformer" and strategy == "temperature":
        from transformer_temperature import PoemTransformer, PoemGenerator_Temperature
        model = PoemTransformer(vocab_size)
        model_path = r"D:/Desktop/AiPoem/model/poem_transformer.pth"
        generator = PoemGenerator_Temperature(token2idx, idx2token, model, device)

    elif model_type == "transformer" and strategy == "top_k":
        from transformer_top_k import PoemTransformer, PoemGenerator_TopK
        model = PoemTransformer(vocab_size)
        model_path = r"D:/Desktop/AiPoem/model/poem_transformer.pth"
        generator = PoemGenerator_TopK(token2idx, idx2token, model, device)

    elif model_type == "transformer" and strategy == "top_p":
        from transformer_top_p import PoemTransformer, PoemGenerator_TopP
        model = PoemTransformer(vocab_size)
        model_path = r"D:/Desktop/AiPoem/model/poem_transformer.pth"
        generator = PoemGenerator_TopP(token2idx, idx2token, model, device)

    else:
        raise ValueError(f"不支持的模型类型或策略: {model_type} | {strategy}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, generator


# ========== 初始化缓存（只运行一次） ==========
def init_models():
    if _cached:
        return _cached  # 已加载则跳过

    token2idx, idx2token, vocab_size = load_vocab()
    models = {}

    for model_type in ['rnn', 'transformer']:
        for strategy in ['temperature', 'top_k', 'top_p']:
            try:
                model, generator = load_model_and_generator(
                    model_type, strategy, token2idx, idx2token, vocab_size
                )
                models[(model_type, strategy)] = (model, generator)
            except Exception as e:
                print(f"[跳过] 加载失败: {model_type}-{strategy}：{e}")

    _cached['vocab'] = (token2idx, idx2token)
    _cached['models'] = models
    _cached['vocab_size'] = vocab_size
    return _cached


# 模块初始化时立即执行
init_models()


# ========== 统一生成函数 ==========
def generate_poem(
        theme: str,
        length: int = 20,
        model_type: str = "rnn",
        strategy: str = "temperature",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9,
        verbose: bool = False
) -> str:
    if not theme or not re.search(r'[\u4e00-\u9fa5]', theme):
        raise ValueError("主题词必须包含中文")

    if length not in (20, 28):
        raise ValueError("长度必须为 20（五言）或 28（七言）")

    if strategy not in ("temperature", "top_k", "top_p"):
        raise ValueError("采样策略必须是 temperature / top_k / top_p 之一")

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    if not (1 <= top_k <= 100):
        raise ValueError("top_k 必须在 1~100 之间")

    if not (0 < top_p <= 1):
        raise ValueError("top_p 必须在 (0,1] 之间")

    token2idx, idx2token = _cached['vocab']
    model, generator = _cached['models'][(model_type, strategy)]

    poem = generator.generate(
        model=model,
        theme=theme,
        max_len=length,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    # 清洗特殊 token
    poem = re.sub(r'<PAD>|<START>|<END>|<UNK>', '', poem)
    poem = re.sub(r'\s+', '', poem)

    if verbose:
        print(f"[生成完成] 模型: {model_type} | 策略: {strategy} | 长度: {length} | 主题: {theme}")

    return poem
