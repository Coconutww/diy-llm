# CS336 Assignment 1 (basics) 执行计划

> 依据：官方作业文档 `Assignment1_Basics.pdf`（Stanford CS336, Spring 2025, Version 1.0.6）
> 实现载体：Datawhale `diy-llm` 仓库 `coursework/assignment1-basics/`（中文 notebook + 训练脚本）
> 参考教材（做完一块看一块，勿提前看答案）：`weiruihhh/cs336_note_and_hw` chapter1/hw1~hw7

---

## 0. 全局概览
TinyStories txt (2.2GB)
   │  BPE ipynb 训练
   ▼
tokenizer.json
   │  get_train_data.py 把文本转成 token id
   ▼
data.bin (int32 数组)
   │  train.py 读它训练
   ▼
ckpt/epoch_N.pt
   │  model.py 的 __main__ 加载
   ▼
生成文本
![alt text](image.png)![alt text](image-1.png)

**要实现的 4 样东西**（全部 from scratch，禁止用 torch.nn / torch.nn.functional / torch.optim，
仅允许：`torch.nn.Parameter`、容器类 `Module/ModuleList/Sequential`、`torch.optim.Optimizer` 基类）：

| # | 实现内容 | 对应章节 | 我们的教学课次 |
|---|---|---|---|
| 1 | BPE 分词器（训练 + 编码/解码） | PDF §2 | 第1课 |
| 2 | Transformer 语言模型（全部组件） | PDF §3 | 第2课 |
| 3 | 交叉熵损失 + AdamW 优化器 + LR 调度 + 梯度裁剪 | PDF §4 | 第3课 |
| 4 | 训练循环（数据加载 + checkpoint） | PDF §5 | 第3、4课 |

**要运行的 4 件事**：在 TinyStories 上 ① 训练 BPE → ② 全数据集转 token ID → ③ 训练 Transformer LM → ④ 生成文本 + 评估 perplexity。官方还有第 5 件（OpenWebText + leaderboard，diy-llm 不含，选做，简历加分项）。

**官方基准配置**（17M 参数 mini-GPT）：vocab_size=10000, context_length=256, d_model=512,
d_ff=1344, theta=10000, 4层16头, 总训练 token 数 ≈ 327.68M（1 张 H100 约 30-40 分钟）。

---

## 1. §2 BPE 分词器 ——【第1课】
思路：任意文本 → 转 UTF‑8 字节流`"牛".encode("utf‑8")` → 在*字节*上跑 BPE 合并算法 → 得到分词器；文本编码输出整数 token id
- 直接拿原始 Unicode 码点做分词器不可行：总共有约 15 万个码点，词表太大、很多字符极其罕见。因此先转成**UTF‑8 字节序列**：所有字符最终拆成 0‑255 的字节，基础词表只有 256 个，非常可控。
- 词to utf8不是一个byte!英文 1 字节，日文 / 中文占 3 字节
```
s = "the"
b = s.encode("utf‑8")   # b'the' → bytes对象
print(list(b))          # [116, 104, 101]
```

### 1.1 Unicode 基础（书面题，2 个 Problem，共 4 分）

| 做什么 | 知识点 |
|---|---|
| unicode1 (1pt)：`chr(0)` 返回什么字符？`__repr__()` 与 print 表现有何不同？该字符出现在文本中会发生什么？ | Unicode 码点（codepoint）、NUL 控制字符、repr vs str 显示差异 |
| unicode2 (3pt)：给出一个"编码后字节序列"的例子并解释 | UTF-8 变长编码规则：ASCII 1 字节 / 常见汉字 3 字节 / emoji 4 字节 |
sol: chr(0) 返回 Unicode 码点 U+0000 的 NUL 字符。它是长度为 1 的不可见字符。print() 输出字符本身，因此通常看不见；__repr__() 则用 \x00 转义表示，便于调试。NUL 可以出现在 Python 字符串中，编码后在 UTF-8 中对应字节 0x00，但在某些 C 风格字符串处理中可能被当作结束标志。
### 1.2 train_bpe（15 分，核心）
- 训练流程
1. 在字节序列统计所有相邻字节对的出现频率
2. 把出现**最高频的字节对**合并，分配一个新 token id 加入词表
3. 反复迭代合并。高频连续字节组合变成一个子词 token。<br>
`run_train_bpe(input_path, vocab_size, special_tokens) → (vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]])`

| 步骤 | 做什么 | 知识点 |
|---|---|---|
| 初始化词表 | 256 个单字节为初始词表，再加入 special_tokens（去重，如 `"a"` 与字节 b'a' 冲突） | **byte-level BPE**：从字节出发 → 无 OOV、覆盖所有语言 |
| 预分词 | 粗粒度先切分文本避免遍历，按 special token 分割（**不允许跨文档边界合并**），再用 GPT-2 正则 PAT 切成 word | low:5，lower:2，widest:3，newest:6; 字节组：low     → (l, o, w)          count=5；lower   → (l, o, w, e, r)    count=2；widest  → (w, i, d, e, s, t) count=3；newest  → (n, e, w, e, s, t) count=6；GPT-2 正则 |
| 统计 | `dict[tuple[bytes], int]` 词频表；一次性统计全部相邻 pair 频次 | lo:7, ow:7, we:8, er:2, wi:3, id:3, de:3, es:9, st:9, ne:6, ew:6 |
| 主循环 | while 词表未满：取最高频 pair → 平局取**字典序更大**者（保证确定性）→ 合并 → 新 token 进词表 | `st > es`，所以合并 `(s,t)` → 新 token `st`，BPE 贪心合并；确定性 |
| 性能关键 | 合并后**只增量更新受影响的 token 的 pair 计数**，不全局重扫（官方点名：naive 实现太慢；此步在 Python 中不可并行） | 增量计数缓存；复杂度分析 |
| 并行加速 | 预分词用 `multiprocessing` 多进程；chunk 边界必须在特殊 token 开头 | 大文件分块处理、边界不切断特殊 token（与"内存监控/资源核算"呼应，diy-llm notebook 里的 find_chunk_boundaries 就是这个） |
| 调优工具 | 用 cProfile/scalene profile 找瓶颈 | 性能剖析 |

### 1.3 用训练好的 BPE 做实验（4 分）

| 做什么 | 知识点 | 资源要求 |
|---|---|---|
| TinyStories 训练 vocab=10000 + `<|endoftext|>`；记录耗时/内存；看最长 token 是否合理；profile 最耗时环节 | 词表大小权衡；语料性质对 merges 的影响 | ≤30min，≤30GB RAM，**无 GPU** |
| OpenWebText 训练 vocab=32000；与 TinyStories 版对比 | 不同领域语料 → 不同合并 | ≤12h，≤100GB RAM（**选做**） |

### 1.4 Tokenizer 类：编码/解码（15 分）

| 做什么 | 知识点 |
|---|---|
| 编码三步：① 特殊 token 分割（按长度降序放进正则，防 `<\|eot\|>` 与 `<\|eot\|><\|eot\|>` 误切）② GPT-2 正则预分词 ③ 对每个 word 按 **merges 创建顺序**贪心应用合并 | merges 顺序 = 编码优先级；**为什么最早合并的优先**（后面的 merge 依赖前面的产物） |
| 分块编码大文件：内存复杂度 O(1)，且**保证 token 不跨 chunk 边界** | 流式处理、内存核算 |
| 解码：id → bytes 拼接 → `decode("utf-8", errors="replace")`（非法字节 → U+FFFD） | 字节级词表的解码鲁棒性 |
### 1.5 Questions
- Q1：为什么从 256 个字节初始化词表？:
  1. 无损覆盖所有语言——任何字符都能用 UTF-8 编码成 0-255 的字节序列,所以永远没有 OOV；
  2. 词表基数极小——只有 256
  个起点，合并搜索空间小；
  3. 空格、标点也是字节，显式建模，不需要专门的词尾标记（官方 PDF
  原文特意加脚注说了这点）。

  ---
- Q2：GPT-2 预分词正则为什么空格要"黏"在词前面
    1. 关键设计：BPE 的合并只在每个 chunk 内部进行，绝不跨  chunk。如果空格被单独切成一个 chunk，那么 cat（句子中间的 cat）和  cat（跟在空格后的 cat）会被拆成同样的字节去合并——模型就分不清"这个词前面有 没有空格"了。把空格黏在词首，让这两种状态成为不同的 token（cat vs  cat），信息不丢失。

  2. 答案代码的 vocab 里存的是真实的空格字节 0x20，不是  Ġ。Ġ 只是 HuggingFace tokenizers 库的显示层做法——直接打印  b' cat'
  你分不清前面的空格，所以显示成 Ġcat
  
  ---
- Q4：编码时为什么选"最早合并"的 pair 优先？
    1. merge #1: (s, t) → "st"     ← 作用在单字节上
  merge #2: (e, "st") → "est" ← 依赖 #1 的产物 "st" 存在！
  merge #3: ("est", "er")...

  BPE 的合并是分层构建的：后面的 merge 建立在前面 merge 的产物之上。
  所以编码时必须按训练时的同一顺序回放：先应用最早的
  merge，它把单字节拼出中间产物（如 st），后面的 merge 才有东西可匹配。

  反例验证：对 newest 的字节 (n,e,w,e,s,t)，如果先去找 merge #2 的 (e,
  st)——序列里根本没有 st 这个元素，永远匹配不上，编码就错了。只有先做
  (s,t)→st，(e,st) 才可能发生。

  面试一句话："编码是训练的回放，顺序错了中间产物就不存在。"

  ---
- Q5：特殊 token 为什么按长度降序放正则？（你答：不知道）

  因为正则的 | 分支是从左到右短路匹配的：在某个位置，先试排在前面的分支，成
  功就赢，后面的分支根本没机会。

  反例：假设特殊 token 是 "ab" 和 "abc"，文本是 xabcx。

  长度降序:   pattern = "abc|ab" → split 出 ["x", "abc", "x"]  ✅
  不排序:     pattern = "ab|abc" → 在 a 处先匹配 "ab"，剩 "c"
             → split 出 ["x", "ab", "cx"]  ❌  "abc" 永远无法整体命中

  对应代码里的场景：如果特殊 token 互相包含（如 <|eot|> 和
  <|eot|><|eot|>），不排序的话长的那个永远匹配不到，会被切成两个短的
---

## 2. Train.py
![alt text](image-2.png)
![alt text](image-3.png)

## §3 Transformer 语言模型 ——【第2课】
$x_1= x + \text{Attention}(\text{RMSNorm}(x))$
$x_{\text{out}}= x_1 + \text{FFN}(\text{RMSNorm}(x_1))$
### 2.0 前置工具与约定

- **强烈推荐 einsum / einops**：任意 batch-like 维度自动处理、自带形状文档（PDF 用一整节讲）
- **数学约定**：论文用列向量 y=Wx，但 PyTorch 行主序 → 代码里写 `x @ W.T`；参数矩阵存 `W`（shape 与 nn.Linear 一致）
- **参数初始化**：Linear 权重 N(0, σ²=2/(d_in+d_out)) 截断 ±3σ；Embedding N(0,1) 截断 ±3；RMSNorm 权重初始化 1；用 `torch.nn.init.trunc_normal_`

### 2.1 组件清单（按实现顺序）

| Problem | 分数 | 做什么 | 知识点 | diy-llm 对应 |
|---|---|---|---|---|
| linear | 1 | Linear 类（无 bias，接口同 nn.Linear） | 现代 LLM 不用 bias；nn.Parameter 注册 | Transformer.ipynb cell 2 |
| embedding | 1 | Embedding 类 | 查表 = 按索引取权重矩阵行 | cell 4 |
| rmsnorm | 1 | RMSNorm | 只缩放不平移；为什么 LLM 用 LayerNorm/RMSNorm 而不用 BatchNorm | cell 7 |
| positionwise_feedforward | 2 | SwiGLU FFN：SiLU 激活 + 门控 | GLU 门控机制；d_ff≈8/3·d_model 且取 64 倍数（GPU 对齐） | cell 10 |
| rope | 2 | RotaryPositionalEmbedding(theta, d_k, max_seq_len)；forward(x, token_positions)，**容忍任意 batch 维度**；cos/sin 预计算为 buffer 按位置索引 | RoPE 旋转矩阵；相对位置如何进入点积；theta 控制频率 | cell 12 |
| softmax | 1 | 沿指定维度、**减 max 保数值稳定** | 溢出与 NaN；softmax 对常数平移不变 | cell 15 |
| scaled_dot_product_attention | 5 | Q,K: (...,L,d_k)，V: (...,L,d_v)；可选 bool mask（True=可 attend）；False 位置**加 -inf** | 缩放 √d_k 的原因；mask 用 -inf 而非 0 的原因；3/4 阶张量测试 | cell 17 |
| multihead_self_attention | 5 | d_k=d_v=d_model/h；**因果 mask**（torch.triu）；Q、K、V 投影共 3 次矩阵乘；**RoPE 只作用于 Q、K 不作用于 V**；head 维度当 batch 维度处理 | 为什么 RoPE 不动 V；信息泄漏问题（看未来 token）；头切分方式 | cell 19 |
| transformer_block | 3 | **pre-norm**：y = x + MHA(RMSNorm(x))，z = y + FFN(RMSNorm(y)) | pre-norm vs post-norm（第5课消融实验伏笔）；残差连接 | cell 21 |
| transformer_lm | 3 | Embedding → num_layers 个 Block → 输出层（logits over vocab） | 预训练目标：下一个 token 预测 | cell 21 + train.py |

### 2.2 transformer_accounting（5 分，书面，**简历面试加分重点**）

- FLOPs 核算：m×n 乘 n×p 的矩阵乘 = 2mnp FLOPs（n 次乘 + n 次加 × mp 个元素）
- 任务：列出 GPT-2 XL（vocab 50257, ctx 1024, 48层, d_model 1600, 25头）前向的所有矩阵乘及 FLOPs
- 给出各模型组件的参数量分解 + 内存构成（参数/激活/梯度/优化器状态）

---

## 3. §4 损失与优化器 ——【第3课】

| Problem | 分数 | 做什么 | 知识点 |
|---|---|---|---|
| cross_entropy | — | logits → 逐 token 损失（diy-llm 用 numpy 实现版） | log-sum-exp 数值稳定技巧；语言模型损失 = 全部位置的平均 |
| learning_rate_tuning | 1 | 调 lr（§7 实验部分执行） | lr 与收敛/发散 |
| adamw | 2 | AdamW 优化器（继承 torch.optim.Optimizer 基类） | **weight decay 与 L2 解耦**；一阶/二阶动量 m、v；**bias correction**；β₁=0.9, β₂=0.95, ε=1e-8 |
| adamwAccounting | 2 | AdamW 资源核算书面题：参数/激活/梯度/优化器状态各多少字节，形如 a·batch_size+b 的显存表达式；GPT-2 XL 训练要多少天 | 优化器状态 = 参数量的 2 倍（m、v 各一份 fp32）——显存大头 |
| learning_rate_schedule | 1 | cosine 调度 + warmup | 为什么 warmup；cosine 衰减公式 |
| gradient_clipping | 1 | 全局 L2 范数裁剪 | loss spike / 梯度爆炸；裁剪是"方向不变、步长受限" |

---

## 4. §5 训练循环 ——【第3、4课】

| Problem | 分数 | 做什么 | 知识点 |
|---|---|---|---|
| data_loading | 2 | `get_batch(x, batch_size, context_length, device)` → (inputs, targets)，形状 (B,L)，"给定前文预测下一个 token" | 随机采样起始点；target = input 右移一位；`np.memmap` 懒加载大文件（**官方点名：数据集放不进内存时用 mmap**） |
| checkpointing | 1 | save/load：model.state_dict + optimizer.state_dict + iteration，`torch.save/load` | 为什么必须存优化器状态（动量）+ 迭代数（恢复 LR 调度） |
| training_together | 4 | 完整训练脚本：CLI 超参、memmap 读数据、定期 eval 日志（wandb） | 训练循环结构：forward → loss → backward → clip → step |

---

## 5. §6 文本生成 ——【第4课】

| Problem | 分数 | 做什么 | 知识点 |
|---|---|---|---|
| decoding | 3 | 取最后位置 logits → softmax → 采样 → 拼回 prompt 循环生成，遇 `<|endoftext|>` 停止 | 自回归采样；**temperature / top-p** 对生成质量的影响；概率分布采样 vs argmax |

---

## 6. §7 实验 ——【第5课】（每项都有官方给的 H100 小时预算）

| Problem | 分数/预算 | 做什么 | 知识点 |
|---|---|---|---|
| experiment_log | 3pt | 日志基础设施：loss 曲线 vs 步数/墙钟时间 | wandb |
| learning_rate | 3pt / 4h | lr sweep → 目标 **val loss ≤1.45**（CPU 降级：40M tokens、≤2.00）；找"发散边界"，讨论 edge of stability | lr 搜索策略；发散现象 |
| batch_size_experiment | 1pt / 2h | batch size 从 1 扫到显存上限，对比 loss 曲线 | batch size 与效率/稳定性的权衡 |
| generate | 1pt | 生成 ≥256 token 文本 + 评流利度 + 至少两个影响因素 | 采样参数调优 |
| layer_norm_ablation | 1pt / 1h | 去掉全部 RMSNorm 训练，原 lr 会怎样？降 lr 能否救回？ | LayerNorm 对训练稳定性的作用 |
| pre_norm_ablation | 1pt / 1h | 改 post-norm（x+FFN 后再归一化）对比 | pre-norm 为什么成为主流 |
| rope_ablation | 1pt / 1h | RoPE vs NoPE 对比 | 位置信息的作用（**diy-llm 没有此项，可自加**） |
| swiglu_ablation | 1pt / 1h | SwiGLU vs 纯 SiLU FFN 对比 | 门控机制的作用 |
| main_experiment | 2pt / 3h | OpenWebText 上训练 + 生成（**选做**） | 更大语料 |
| leaderboard | 6pt / 10h | 提交 OWT 验证 loss（**选做**） | — |

---

## 7. 官方 vs diy-llm 差异（注意点）

| 维度 | 官方 PDF | diy-llm（我们的载体） |
|---|---|---|
| 代码组织 | cs336_basics 包 + adapters.py + pytest 测试 | 2 个 notebook + get_train_data.py / train.py / model.py |
| 词表 | TinyStories 用 vocab=10000 | train.py 默认 vocab=50257（GPT-2 词表，直接用 notebook 训出的 tokenizer） |
| 消融 | 4 个（含 RoPE vs NoPE） | 3 个（SiLU / no_RMSNorm / post_Norm），**建议自己补 RoPE-NoPE** |
| 资源核算书面题 | unicode1/2、transformer_accounting、adamwAccounting | notebook 里的 Q&A cell 部分覆盖 |
| OWT + leaderboard | 有 | 无（**选做**，简历最强加分项） |
| 调试技巧 | 官方建议：先 overfit 单个 minibatch 到 loss≈0；断点查形状；监控激活/权重/梯度范数 | 同样适用 |

---

## 8. 执行路线（与 6 课教学对应）

```
第1课 BPE（§2）            → 本地 CPU，无 GPU 需求
第2课 Transformer 组件（§3）→ 本地 CPU（测试全可跑）
第3课 损失/优化器（§4）+ 数据加载/checkpoint（§5）→ 本地 CPU
第4课 生成（§6）+ 整机训练（§5.3）→ ★ 需要 GPU（AutoDL 3090）
第5课 实验与消融（§7）       → ★ 需要 GPU（预算内约 2-4 小时 3090）
第6课 模拟面试拷打          → 覆盖：BPE / 组件原理 / 训练细节 / 资源核算 / 实验结论
```

**GPU 边界**：第1-3课全部在本地完成（包括 notebook 全部测试 cell 与 BPE 真实训练，
官方要求 ≤30min/30GB RAM 无 GPU）。第4课起上 AutoDL。

**调试军规（官方建议，全程适用）**：
1. 模型搭好后先 overfit 单个 minibatch（loss 应能迅速逼近 0），证明架构正确
2. 在组件里设断点检查中间张量形状
3. 监控激活/权重/梯度的范数，防爆炸/消失
