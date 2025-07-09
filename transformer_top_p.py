import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
from typing import List, Dict

# ====================== 1. 加载词表 ======================
def load_vocab(file_path: str) -> Dict[str, int]:
    """加载词表映射文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ====================== 2. 定义 Transformer 模型 ======================
class PoemTransformer(nn.Module):
    """Transformer模型（保持原结构，确保兼容性）"""
    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=6, ff_dim=512, dropout=0.1):
        super(PoemTransformer, self).__init__()
        
        # 词向量层
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)  # 主程序中修正padding_idx
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.rand(1, 5000, embed_size))
        
        # 完整Transformer结构
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        """生成自回归掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, tgt):
        # 嵌入和位置编码
        src = self.dropout(self.embed(src) + self.positional_encoding[:, :src.size(1), :])
        tgt = self.dropout(self.embed(tgt) + self.positional_encoding[:, :tgt.size(1), :])
        
        # 生成掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # 通过Transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        return self.fc_out(output), None

# ====================== 3. 古诗生成器（支持押韵+五言/七言+Top-P采样） ======================
class PoemGenerator_TopP:
    def __init__(self,
                 token2idx: Dict[str, int],
                 idx2token: Dict[int, str],
                 model: torch.nn.Module,
                 device: str = "cpu"):

        self.token2idx = token2idx
        self.idx2token = idx2token
        self.model = model.to(device)
        self.device = device
        self.associations = self._load_associations("word_associations.json")
        
        # 押韵字典（简化版平水韵，按韵母分类）
        self.rhyme_dict = self._build_rhyme_dict()
        
        # 初始化分词器
        try:
            import thulac
            self.thulac = thulac.thulac(seg_only=True)
        except ImportError:
            print("警告: 未安装THULAC分词器，将使用简单分词方法")
            self.thulac = None

    def _build_rhyme_dict(self) -> Dict[str, str]:
        """构建简化版押韵字典（按现代拼音韵母分类）"""
        rhyme_dict = {}
        # 基础韵部（覆盖常用字）
        rhyme_categories = {
            'a': ['花', '家', '霞', '沙', '茶', '麻', '涯', '瓜', '华', '芽', '佳', '斜'],
            'o': ['歌', '多', '河', '波', '罗', '柯', '戈', '磨', '蓑', '荷', '婆'],
            'i': ['衣', '期', '池', '知', '时', '丝', '诗', '棋', '词', '啼', '溪', '西'],
            'u': ['无', '图', '湖', '孤', '壶', '途', '苏', '书', '珠', '浮', '奴'],
            'an': ['山', '间', '闲', '还', '颜', '关', '湾', '环', '班', '丹', '残', '天'],
            'ang': ['长', '香', '光', '阳', '堂', '芳', '昌', '央', '刚', '桑', '忙'],
            'eng': ['风', '声', '灯', '僧', '升', '生', '星', '耕', '更', '情', '城'],
            'ong': ['东', '同', '中', '空', '红', '通', '工', '蓬', '浓', '松', '龙']
        }
        for rhyme, chars in rhyme_categories.items():
            for char in chars:
                rhyme_dict[char] = rhyme
        return rhyme_dict

    def _load_associations(self, path: str) -> Dict:
        """加载关联词库"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"警告: 未找到关联词库文件 {path}，将使用空字典。")
        return {}

    def segment_with_thulac(self, text: str) -> List[str]:
        """分词方法"""
        if self.thulac:
            try:
                result = self.thulac.cut(text)
                return [word for word, _ in result]
            except:
                pass
        return list(text)  # 回退到单字分词

    def get_associated_words(self, word: str, top_k: int = 5) -> List[str]:
        """获取输入词的关联词列表"""
        if word not in self.associations:
            return []
        related_words = sorted(
            self.associations[word].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [w for w, _ in related_words[:top_k]]

    def generate(
        self,
        model: PoemTransformer,
        theme: str,
        max_len: int = 20,
        strategy: str = "top_p",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9
    ) -> str:
        """
        生成古诗，支持押韵、句式切换和Top-P采样
        top_p: Nucleus采样参数，累积概率阈值（0-1之间）
        """
        # 1. 确定句式参数
        chars_per_line = 5 if max_len == 20 else 7
        lines_count = 4  # 绝句固定4句
        total_chars = chars_per_line * lines_count  # 总字数
        
        # 2. 分词与联想
        segmented_words = self.segment_with_thulac(theme)
        print(f"分词结果: {segmented_words}")
        
        # 收集候选词
        all_candidates = []
        for word in segmented_words:
            all_candidates.append(word)
            all_candidates.extend(self.get_associated_words(word, top_k=5))
        
        # 过滤有效词，补充主题相关常用字
        valid_words = list(set(w for w in all_candidates if w in self.token2idx))
        if not valid_words:
            valid_words = [w for w in list(self.token2idx.keys())[:10] if w not in ['<PAD>', '<START>']]
        
        # 根据主题补充候选词
        theme_related = {
            '春天': ['春', '风', '花', '柳', '燕', '啼', '暖', '芽'],
            '秋天': ['秋', '霜', '叶', '月', '雁', '寒', '枫', '露'],
            '夏天': ['夏', '荷', '蝉', '雨', '荫', '热', '蛙', '莲'],
            '冬天': ['冬', '雪', '寒', '梅', '冰', '霜', '风', '松']
        }.get(theme, ['日', '云', '山', '水', '天', '地', '人', '心'])
        
        valid_words.extend([w for w in theme_related if w in self.token2idx and w not in valid_words])
        valid_words = list(set(valid_words))  # 去重
        print(f"有效候选词: {valid_words}")

        # 3. 选择韵部（随机选一个包含候选词的韵部）
        candidate_rhymes = set()
        for word in valid_words:
            if word in self.rhyme_dict:
                candidate_rhymes.add(self.rhyme_dict[word])
        if not candidate_rhymes:
            candidate_rhymes = set(self.rhyme_dict.values())  # 兜底：用所有韵部
        target_rhyme = random.choice(list(candidate_rhymes))  # 目标韵部
        rhyme_words = [w for w in valid_words if self.rhyme_dict.get(w) == target_rhyme]
        if not rhyme_words:
            rhyme_words = [w for w in self.rhyme_dict if self.rhyme_dict[w] == target_rhyme]  # 兜底
        print(f"目标韵部: {target_rhyme}, 押韵候选字: {rhyme_words[:5]}")


        input_tokens = []
        for word in segmented_words:
            if word in self.token2idx:
                input_tokens.append(self.token2idx[word])

        # 如果输入为空，使用默认起始词
        if not input_tokens:
            input_tokens = [self.token2idx[valid_words[0]]] if valid_words else [self.token2idx['<START>']]
            
        
        # 确保不超过总字数限制
        input_tokens = input_tokens[:total_chars]
        generated = [self.idx2token[idx] for idx in input_tokens]

        # 5. 生成诗句（核心逻辑：使用Top-P采样）
        self.model.eval()
        with torch.no_grad():
            src = torch.tensor([input_tokens]).to(self.device)
            tgt = src.clone()
            
            while len(generated) < total_chars:
                current_total = len(generated)
                current_line = current_total // chars_per_line  # 0-3
                current_pos_in_line = current_total % chars_per_line  # 0到chars_per_line-1

                output, _ = self.model(src, tgt)
                next_logits = output[:, -1, :]  # 最后一个位置的预测
                
                # Top-P采样实现
                # 1. 应用温度调整
                next_logits = next_logits / temperature
                # 2. 计算softmax概率
                probs = F.softmax(next_logits, dim=-1)
                # 3. 排序概率和索引
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # 4. 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 5. 找到满足累积概率 <= top_p 的截断点
                cutoff = (cumulative_probs <= top_p).sum().item()
                # 6. 确保至少保留一个候选词
                cutoff = max(cutoff, 1)
                # 7. 过滤候选词
                filtered_indices = sorted_indices[0, :cutoff]
                filtered_probs = sorted_probs[0, :cutoff]
                # 8. 重归一化概率
                filtered_probs = filtered_probs / filtered_probs.sum()
                # 9. 采样选择
                next_idx = filtered_indices[torch.multinomial(filtered_probs, 1).item()].item()
                
                next_word = self.idx2token.get(next_idx, '')

                # 过滤无效字符
                invalid_tokens = ['<END>', '<UNK>', '<PAD>', '<START>']
                if next_word in invalid_tokens:
                    continue
                
                # 添加到结果
                generated.append(next_word)
                input_tokens.append(next_idx)
                tgt = torch.tensor([input_tokens]).to(self.device)

        # 6. 格式化输出（按句式分割）
        poem = "".join(generated)
        for token in invalid_tokens:
            poem = poem.replace(token, "")
        
        # 按句分割
        lines = []
        for i in range(0, min(len(poem), total_chars), chars_per_line):
            line = poem[i:i + chars_per_line]
            if len(line) == chars_per_line:
                lines.append(line)

        lines = self._add_punctuation(lines)


        return "\n".join(lines[:lines_count])

    def _add_punctuation(self, lines: List[str]) -> List[str]:
        """根据格式添加标点符号（交替使用逗号和句号）"""
        for i in range(len(lines)):
            if i == 0 or i == 2:
                lines[i] += "，"
            else:
                lines[i] += "。"
        return lines


# ====================== 4. 模型加载与主程序 ======================
def load_model(vocab_size, model_path=None, device='cpu'):
    """加载模型并处理权重兼容问题"""
    model = PoemTransformer(vocab_size).to(device)
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用随机初始化权重")
    else:
        print("警告: 未找到预训练模型，将使用随机初始化权重")
    return model