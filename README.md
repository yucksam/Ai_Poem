# 基于深度学习的古诗生成系统

本项目致力于构建一个融合中华传统文化与现代人工智能技术的古诗自动生成系统，支持用户根据起始词、模型类型和采样策略生成风格多样的律诗。

---

## 🌟 项目亮点

- ✍️ **双模型支持**：基于 RNN（LSTM）和 Transformer 的字符级古诗生成模型
- 🔥 **多策略采样**：支持 Temperature、Top-K、Top-P 等采样方式
- 📈 **模型性能优越**：Transformer 模型困惑度约 92，RNN 模型约 97
- 🧠 **押韵与平仄支持**：构建韵部字典，提升诗句合律性
- 🌐 **完整 Web 界面**：前后端完整部署，便于交互体验

---

## 📁 项目结构

```
AiPoem/
├── app.py
├── generator.py
├── rnnmodel.py
├── transformer_temperature.py
├── transformer_top_k.py
├── transformer_top_p.py
├── word_associations.json
├── templates/
│   └── index.html
├── static/
│   └── bd.jpeg
├── data/
│   └── chinese-poetry/
├── model/
│   ├── idx2token.json
│   ├── token2idx.json
│   ├── tokenized_poems.json
│   ├── poem_rnn.pth
│   ├── poem_transformer.pth
│   ├── rnnmodel.ipynb
│   ├── rnntrain.ipynb
│   ├── transformermodel.ipynb
│   └── transformertrain.ipynb
```

---

## ⚙️ 使用方式

### 🔹 启动服务

```bash
pip install -r requirements.txt
python app.py
```

访问：[http://localhost:5000](http://localhost:5000)

### 🔹 API 调用

- 请求地址：`/generate_poem`（POST）
- 参数示例：

```json
{
  "start": "秋风",
  "length": 28,
  "model": "transformer",
  "sampling": "top_k",
  "top_k": 8
}
```

---

## 🎯 参数设置说明

| 参数名        | 类型   | 默认值          | 说明                                  |
| ------------- | ------ | --------------- | ------------------------------------- |
| `start`       | string | 必填            | 起始词（生成开头）                    |
| `length`      | int    | `20`            | 总字数，20=五言，28=七言              |
| `model`       | string | `"rnn"`         | 可选：`rnn`、`transformer`            |
| `sampling`    | string | `"temperature"` | 可选：`temperature`、`top_k`、`top_p` |
| `temperature` | float  | `1.0`           | 热度采样参数（>1 更随机）             |
| `top_k`       | int    | `10`            | Top-K 采样阈值                        |
| `top_p`       | float  | `0.9`           | Top-P 采样阈值                        |

---

## 🧠 模型训练说明

### RNN 模型

- 📄 文件路径：`model/rnntrain.ipynb`
- 📌 说明：
  - 使用 LSTM 结构，Embedding 维度为 128，隐藏层为 256。
  - 训练语料为 `tokenized_poems.json`。
  - 使用交叉熵损失函数（`nn.CrossEntropyLoss`）与 Adam 优化器。
  - 模型保存路径：`model/poem_rnn.pth`

### Transformer 模型

- 📄 文件路径：`model/transformertrain.ipynb`
- 📌 说明：
  - 使用 PyTorch 的标准 `nn.Transformer` 构建 6 层编码器+解码器结构。
  - 每层包含 8 个注意力头，FeedForward 网络维度为 512。
  - 使用位置编码、掩码、词嵌入进行训练。
  - 训练使用的文件：`tokenized_poems.json`、词表文件 `token2idx.json` / `idx2token.json`
  - 模型保存路径：`model/poem_transformer.pth`

## 🚀 训练注意事项

- 建议使用 GPU 运行，训练 Transformer 模型时更显著。
- 所有模型训练使用 `tokenized_poems.json` 和词表 `token2idx.json`、`idx2token.json`
- 训练前请先运行分词与词表构建代码（可参考：`rnnmodel.ipynb` 与 `transformermodel.ipynb`）。
- 训练时间因设备和数据量而异，一般需数十分钟至数小时。
- 分词建议使用 `jieba` 或 `thulac`（可选）

---

如需重训模型，请确保：

- 安装好所有依赖
- 准备好 `data/chinese-poetry/` 下的原始古诗数据
- 确保 `model/` 目录有写权限用于保存 `.pth` 模型文件

---

## 📦 依赖安装（requirements.txt）

```txt
torch>=1.12.1
flask
jieba
opencc-python-reimplemented
thulac  # 可选，用于更精细的中文分词
```

安装命令：

```bash
pip install -r requirements.txt
```

---

## 🔧 TODO

- [ ] 接入 GPT 模型提升质量
- [ ] 增加用户风格/情绪控制
- [ ] 实现古文注释与现代翻译功能
- [ ] 增强移动端适配能力

---

## 📜 致谢

- 数据来源：[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- 项目成员：周颖、江子怡、张范渝超、刘子阳、鲜诗颖、郭海枫

> “千年诗意，一念即生” —— 愿技术之笔，助文化之光。
