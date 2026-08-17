# Mini-Transformer-ja-en

基于 Transformer 架构的轻量级日英机械翻译。
完整实现了 Transformer 以及自定义训练评估流水线。

---


## 训练与评估

* **数据集**：tatoeba和opus各取一点，组成40w的平行语料库
* **评估指标**：BLEU-4 Score
* **最终成绩**：`0.477`
* **训练日志与权重**：各阶段 checkpoint 及验证曲线详见 `result/` 目录。

> **注**：尽管 BLEU 指标表现优异，但在复杂长难句及前置定语嵌套（如主语实体绑定）场景中，自回归模型仍存在局部上下文衰减的现象。这也为迭代版本提出了改进方向。

---

## 项目结构

```
data/                  # 预处理数据与 Tokenizer 序列化文件 (.pkl)
models/                # Transformer 核心网络模块
result/                # 训练过程产物 (模型权重 .pth、Loss 曲线图、评估日志)
train.ipynb            # 模型训练与推理交互脚本
README.md              # 项目说明文档   
```