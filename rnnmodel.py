import torch
import torch.nn as nn
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ========= 神经网络模型定义 =========
class PoemRNN(nn.Module):
    """
    基于 LSTM 的古诗生成模型（Embedding + 多层 LSTM + 线性输出）
    """
    def __init__(self, vocab_size: int, embed_size: int=256, hidden_size: int=256, num_layers: int=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # 字嵌入层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM 层
        self.fc = nn.Linear(hidden_size, vocab_size)  # 输出为词表大小（每个字的概率）

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        前向传播
        :param x: 输入字索引序列 (batch_size, seq_len)
        :param hidden: 上一个时间步的隐藏状态 (h, c)
        :return: 输出 logits (batch, seq_len, vocab_size) 和新的 hidden 状态
        """
        x = self.embed(x)  # 嵌入为向量
        output, hidden = self.lstm(x, hidden)  # 通过 LSTM
        output = self.fc(output)  # 映射回词表空间
        return output, hidden


# ========= 古诗生成器类 =========
class EnhancedPoemGenerator:
    """
    古诗生成核心类，集成模型调用、采样、排版、押韵与平仄修正等能力
    """
    def __init__(self, token2idx: Dict[str, int], idx2token: Dict[int, str], device: str='cpu'):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.device = device

        # 初始化辅助资源
        self.rhyme_groups = self._init_rhyme_groups()  # 押韵字库
        self.pingze_map = self._init_pingze_map()      # 平仄映射
        self.thesaurus = self._init_thesaurus()        # 同义词替换表
        self.meter_templates = self._init_meter_templates()  # 格律模板

    def generate(
        self,
        model: PoemRNN,
        theme: str,
        max_len: int = 20,
        strategy: str = "temperature",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9
    ) -> str:
        """
        主调度方法：调用模型生成、格式化行、平仄和押韵修正、加标点
        :param model: 已加载的 PoemRNN 模型
        :param theme: 起始字或词
        :param max_len: 总字数（通常 20 或 28）
        :param strategy: 采样策略（temperature/top_k/top_p）
        :param temperature: 温度采样参数
        :param top_k: top-k 采样参数
        :param top_p: top-p 采样参数
        :return: 最终格式化后的诗歌文本
        """
        raw_text = self._generate_raw_sequence(model, theme, max_len, strategy, temperature, top_k, top_p)
        lines = self._format_to_lines(raw_text, max_len)
        lines = self._adjust_rhyme(lines)
        lines = self._adjust_pingze(lines)
        return self._add_punctuation(lines)

    def _generate_raw_sequence(self, model, theme, max_len, strategy, temperature, top_k, top_p):
        """
        使用指定采样策略生成原始字序列
        """
        input_idx = torch.tensor([[self.token2idx.get(theme, self.token2idx['<UNK>'])]]).to(self.device)
        hidden = None
        generated = [theme]

        for _ in range(max_len - 1):
            with torch.no_grad():
                output, hidden = model(input_idx, hidden)

            logits = output[0, -1]

            # ---------- 支持三种采样策略 ----------
            if strategy == "temperature":
                logits = logits / temperature
                probs = torch.softmax(logits, -1)
                next_idx = torch.multinomial(probs, 1).item()

            elif strategy == "top_k":
                probs = torch.softmax(logits, -1)
                topk_probs, topk_indices = torch.topk(probs, top_k)
                topk_probs = topk_probs / topk_probs.sum()
                next_idx = topk_indices[torch.multinomial(topk_probs, 1).item()].item()

            elif strategy == "top_p":
                probs = torch.softmax(logits, -1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
                cutoff_idx = cutoff[0] + 1 if len(cutoff) > 0 else len(sorted_probs)
                filtered_probs = sorted_probs[:cutoff_idx]
                filtered_indices = sorted_indices[:cutoff_idx]
                filtered_probs = filtered_probs / filtered_probs.sum()
                next_idx = filtered_indices[torch.multinomial(filtered_probs, 1).item()].item()
            else:
                raise ValueError(f"未知采样策略: {strategy}")

            if next_idx == self.token2idx['<END>']:
                break

            generated.append(self.idx2token[next_idx])
            input_idx = torch.tensor([[next_idx]]).to(self.device)

        return "".join(generated)

    def _format_to_lines(self, text: str, max_len: int) -> List[str]:
        """
        将生成的文本按每行字数切分为 4 行
        """
        char_per_line = 5 if max_len == 20 else 7
        cleaned = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return [cleaned[i:i+char_per_line] for i in range(0, min(len(cleaned), 4*char_per_line), char_per_line)][:4]

    def _adjust_rhyme(self, lines: List[str]) -> List[str]:
        """
        调整最后一句与第二句押韵
        """
        if len(lines) >= 2:
            rhyme_char = lines[1][-1]
            rhyme_group = next((k for k, v in self.rhyme_groups.items() if rhyme_char in v), None)
            if rhyme_group and len(lines) > 3:
                last_char = lines[3][-1]
                if last_char not in self.rhyme_groups[rhyme_group]:
                    lines[3] = lines[3][:-1] + random.choice(self.rhyme_groups[rhyme_group])
        return lines

    def _adjust_pingze(self, lines: List[str]) -> List[str]:
        """
        尝试修复每行的平仄格式
        """
        for i, line in enumerate(lines):
            if len(line) == 5:  # 仅处理五言
                pattern = ['仄', '仄', '平', '平', '仄'] if i % 2 == 0 else ['平', '平', '仄', '仄', '平']
                new_line = []
                for j, char in enumerate(line):
                    if j < len(pattern) and self.pingze_map[char] != pattern[j]:
                        candidates = [c for c in self.thesaurus.get(char, [char]) if self.pingze_map.get(c, '仄') == pattern[j]]
                        new_line.append(random.choice(candidates) if candidates else char)
                    else:
                        new_line.append(char)
                lines[i] = "".join(new_line)
        return lines

    def _add_punctuation(self, lines: List[str]) -> str:
        """
        按格式添加标点符号（4句：逗号、句号交替）
        """
        cleaned = [re.sub(r'[\n\r]+', '', line) for line in lines[:4]]
        return (
            f"{cleaned[0]}，"
            f"{cleaned[1]}。"
            f"{cleaned[2]}，"
            f"{cleaned[3]}。"
        )

    # ===================== 资源初始化 =====================
    def _init_rhyme_groups(self) -> Dict[str, List[str]]:
        """人工定义押韵组"""
        return {
            'a': ['花', '家', '华', '霞', '涯', '沙', '茶', '麻', '纱'],
            'o': ['歌', '多', '河', '波', '罗', '梭', '柯', '戈', '磨'],
            'i': ['枝', '时', '丝', '迟', '诗', '知', '痴', '池', '脂'],
            'u': ['无', '图', '湖', '孤', '壶', '途', '酥', '糊', '乌']
        }

    def _init_pingze_map(self) -> Dict[str, str]:
        """部分字的平仄映射，缺失的默认为仄"""
        pingze = defaultdict(lambda: '仄')
        pingze.update({
            '春': '平', '风': '平', '秋': '平', '天': '平', '空': '平',
            '年': '平', '来': '平', '时': '平', '人': '平', '明': '平',
            '月': '仄', '日': '仄', '雪': '仄', '白': '仄', '玉': '仄'
        })
        return pingze

    def _init_thesaurus(self) -> Dict[str, List[str]]:
        """简易同义词词库（用于平仄替换）"""
        # return {
        #     '春': ['春', '风', '花', '晨', '朝'],
        #     '秋': ['秋', '月', '霜', '夕', '夜'],
        #     '山': ['山', '峰', '岳', '岭', '岩'],
        #     '水': ['水', '江', '河', '湖', '海']
        # }
        return {}

    def _init_meter_templates(self) -> Dict[str, Dict]:
        """格律模板，目前未直接使用"""
        return {
            '五言绝句': {
                'patterns': [
                    ['仄', '仄', '平', '平', '仄'],
                    ['平', '平', '仄', '仄', '平']
                ]
            },
            '七言绝句': {
                'patterns': [
                    ['平', '平', '仄', '仄', '平', '平', '仄'],
                    ['仄', '仄', '平', '平', '仄', '仄', '平']
                ]
            }
        }



# import torch
# import torch.nn as nn
# import random
# import re
# from collections import defaultdict
# from typing import Dict, List, Tuple, Optional
#
# class PoemRNN(nn.Module):
#     """完整的RNN模型定义"""
#     def __init__(self, vocab_size: int, embed_size: int=256, hidden_size: int=256, num_layers: int=2):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
#         x = self.embed(x)
#         output, hidden = self.lstm(x, hidden)
#         output = self.fc(output)
#         return output, hidden
#
# class EnhancedPoemGenerator:
#     """完整功能的古诗生成器"""
#     def __init__(self, token2idx: Dict[str, int], idx2token: Dict[int, str], device: str='cpu'):
#         self.token2idx = token2idx
#         self.idx2token = idx2token
#         self.device = device
#
#         # 完整诗词资源
#         self.rhyme_groups = self._init_rhyme_groups()
#         self.pingze_map = self._init_pingze_map()
#         self.thesaurus = self._init_thesaurus()
#         self.meter_templates = self._init_meter_templates()
#
#     def generate(self, model: PoemRNN, theme: str, max_len: int=20) -> str:
#         """
#         完整生成流程（与你现有app.py调用方式完全一致）
#         返回格式：带标点符号和换行的完整古诗
#         """
#         # 1. 生成原始序列
#         raw_text = self._generate_raw_sequence(model, theme, max_len)
#
#         # 2. 规范化处理
#         lines = self._format_to_lines(raw_text, max_len)
#         lines = self._adjust_rhyme(lines)
#         lines = self._adjust_pingze(lines)
#
#         # 3. 添加标点
#         punctuated = self._add_punctuation(lines)
#
#         # 4. 生成标题
#         # title = self._generate_title(theme, lines)
#
#         return punctuated  # 返回完整的古诗文本
#
#     # ========== 以下是完整的内部方法 ==========
#     def _generate_raw_sequence(self, model: PoemRNN, theme: str, max_len: int) -> str:
#         """核心生成逻辑"""
#         input_idx = torch.tensor([[self.token2idx.get(theme, self.token2idx['<UNK>'])]]).to(self.device)
#         hidden = None
#         generated = [theme]
#
#         for _ in range(max_len - 1):
#             with torch.no_grad():
#                 output, hidden = model(input_idx, hidden)
#
#             # 温度采样
#             logits = output[0, -1] / 0.7  # 固定温度0.7
#             probs = torch.softmax(logits, -1)
#             next_idx = torch.multinomial(probs, 1).item()
#
#             if next_idx == self.token2idx['<END>']:
#                 break
#
#             generated.append(self.idx2token[int(next_idx)])
#             input_idx = torch.tensor([[int(next_idx)]]).to(self.device)
#
#         return "".join(generated)
#
#     def _format_to_lines(self, text: str, max_len: int) -> List[str]:
#         """按字数分行"""
#         char_per_line = 5 if max_len == 20 else 7
#         cleaned = re.sub(r'[^\u4e00-\u9fa5]', '', text)
#         return [cleaned[i:i+char_per_line] for i in range(0, min(len(cleaned), 4*char_per_line), char_per_line)][:4]
#
#     def _adjust_rhyme(self, lines: List[str]) -> List[str]:
#         """押韵调整"""
#         if len(lines) >= 2:
#             rhyme_char = lines[1][-1]
#             rhyme_group = next((k for k, v in self.rhyme_groups.items() if rhyme_char in v), None)
#             if rhyme_group and len(lines) > 3:
#                 last_char = lines[3][-1]
#                 if last_char not in self.rhyme_groups[rhyme_group]:
#                     lines[3] = lines[3][:-1] + random.choice(self.rhyme_groups[rhyme_group])
#         return lines
#
#     def _adjust_pingze(self, lines: List[str]) -> List[str]:
#         """平仄调整"""
#         for i, line in enumerate(lines):
#             if len(line) == 5:  # 五言
#                 pattern = ['仄', '仄', '平', '平', '仄'] if i % 2 == 0 else ['平', '平', '仄', '仄', '平']
#                 new_line = []
#                 for j, char in enumerate(line):
#                     if j < len(pattern) and self.pingze_map[char] != pattern[j]:
#                         candidates = [c for c in self.thesaurus.get(char, [char]) if self.pingze_map.get(c, '仄') == pattern[j]]
#                         new_line.append(random.choice(candidates) if candidates else char)
#                     else:
#                         new_line.append(char)
#                 lines[i] = "".join(new_line)
#         return lines
#
#     def _add_punctuation(self, lines: List[str]) -> str:
#         """处理已含换行符的lines输入"""
#         # 1. 彻底清洗换行符
#         cleaned = [re.sub(r'[\n\r]+', '', line) for line in lines[:4]]
#
#         # # 2. 补全行数
#         # while len(cleaned) < 4:
#         #     cleaned.append("〇" * (5 if len(cleaned[0]) == 5 else 7))
#
#         # 3. 硬编码格式（绝对控制）
#         return (
#             f"{cleaned[0]}，"
#             f"{cleaned[1]}。"
#             f"{cleaned[2]}，"    # 单换行
#             f"{cleaned[3]}。"      # 无换行
#         )
#
#
#     # ========== 完整的资源初始化 ==========
#     def _init_rhyme_groups(self) -> Dict[str, List[str]]:
#         return {
#             'a': ['花', '家', '华', '霞', '涯', '沙', '茶', '麻', '纱'],
#             'o': ['歌', '多', '河', '波', '罗', '梭', '柯', '戈', '磨'],
#             'i': ['枝', '时', '丝', '迟', '诗', '知', '痴', '池', '脂'],
#             'u': ['无', '图', '湖', '孤', '壶', '途', '酥', '糊', '乌']
#         }
#
#     def _init_pingze_map(self) -> Dict[str, str]:
#         pingze = defaultdict(lambda: '仄')
#         pingze.update({
#             # 平声字
#             '春':'平', '风':'平', '秋':'平', '天':'平', '空':'平',
#             '年':'平', '来':'平', '时':'平', '人':'平', '明':'平',
#             # 仄声字
#             '月':'仄', '日':'仄', '雪':'仄', '白':'仄', '玉':'仄'
#         })
#         return pingze
#
#     def _init_thesaurus(self) -> Dict[str, List[str]]:
#         return {
#             '春': ['春', '风', '花', '晨', '朝'],
#             '秋': ['秋', '月', '霜', '夕', '夜'],
#             '山': ['山', '峰', '岳', '岭', '岩'],
#             '水': ['水', '江', '河', '湖', '海']
#         }
#
#     def _init_meter_templates(self) -> Dict[str, Dict]:
#         return {
#             '五言绝句': {
#                 'patterns': [
#                     ['仄', '仄', '平', '平', '仄'],
#                     ['平', '平', '仄', '仄', '平']
#                 ]
#             },
#             '七言绝句': {
#                 'patterns': [
#                     ['平', '平', '仄', '仄', '平', '平', '仄'],
#                     ['仄', '仄', '平', '平', '仄', '仄', '平']
#                 ]
#             }
#         }