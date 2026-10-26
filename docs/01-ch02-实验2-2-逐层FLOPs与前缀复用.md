# 实验 2-2 分析：输入行数相同，注意力计算量为何不同

> 练习 2-2〔核心〕：用表 2-1，取 B=4, S=4096, P=1024，分别计算投影与 FFN、有效因果注意力、词表投影（每条序列最后一个 token）的 FLOPs；保持 BP=4096 改为 B=1, S=4096, P=4096，先预测哪些项相同再算差异；最后比较 8K 输入完整处理与复用 6144 前缀（只补 2048 token）两种方式，求节省的矩阵运算量比例。
>
> 正文对应 `manuscripts/02-模型架构.md` 2.1.4 节表 2-1 及其后的汇总公式（仓库中章节在 `manuscripts/`，无 `book/` 目录）。

## 1. 练习要求 → 代码实现映射

| 练习要求 | 实现位置 | 说明 |
| --- | --- | --- |
| 表 2-1 每一行（Q/KV 投影、QK^T 与 AV、输出投影、SwiGLU、词表头） | `calculations/src/infra_calc/models/qwen3.py:77-177` 的 `build_operators()` | 每行表项对应一个 `Operator`，`repeats` 字段就是表中"层／调用数"列 |
| m=BP（输入行数）、N_pair（有效因果配对）、m_out（词表头行数） | `calculations/src/infra_calc/schema.py:26-41` 的 `Scenario.rows / pairs / head_rows` | 练习记号到代码字段的翻译规则，见第 3.1 节 |
| 汇总公式 F=36(385875968·m+16384·N_pair)+2·m_out·4096·151936 | `qwen3.py:200-238` 的 `summarize()` | `backbone_projection_ffn_flops`、`causal_attention_matrix_flops`、`lm_head` 算子三项 |
| 练习场景 1/2（B=4,S=4096,P=1024 与 B=1,S=4096,P=4096） | **不在已复算结果中**，用 `python3 calculations/calc.py forward --batch 4 --history 4096 --tokens 1024` 即时计算 | `cli.py:86-96`；已验证 CLI 与 reproduce 产物逐位一致 |
| 第 3 问的两个场景 | 已复算文件 `qwen3-8b-prefill-8192.json`（S=0,P=8192）与 `qwen3-8b-prefix-6144-plus-2048.json`（S=6144,P=2048） | 实验脚本 `run.py:23-28` 的 `SCENARIOS` 清单 |
| 23 个算子 → 14 个功能组（README 的分组表） | `experiments/ch02/02-02/run.py:30-68` 的 `GROUP` 映射与 `group_rows()` | |

## 2. 数值的产生链路

```
calc.py → cli.py main() ─┬─ "reproduce" → reproduce.run()
                         │     └─ 对 scenarios/book.json 里每个 forward 场景：
                         │        model_forward(model, Scenario(batch, history, tokens, ...))
                         │        → save(id) → results/qwen3-8b-{prefill-8192,decode-b1-s8192,
                         │                                   decode-b64-s8192,prefix-6144-plus-2048}.json
                         └─ "forward" → 直接调 model_forward()，参数来自命令行   ← 练习场景 1/2 用这条
```

`models/__init__.py:8` 的 `forward()` 按官方 config 的 `model_type` 分发，Qwen3-8B 是 dense → `qwen3.calculate()`（`qwen3.py:180`）：读 `configs/models/qwen3-8b/config.json`（经 `sources.read_source()` 校验哈希，该文件在本仓库完好）→ `build_operators()` 构造 23 个算子 → `summarize()` 汇总。

与实验 2-1 不同的一点：**本练习的四个数据文件都在**，且场景 1/2 可以用 `calc.py forward` 现场复算，完全不经过 reproduce（不需要 scenarios/book.json，也不受 LFS 指针问题影响）。已验证：CLI 算出的 133,594,323,353,600 与 37,110,366,076,928 和已复算文件逐位相同。

## 3. 核心代码解析

### 3.1 练习记号 → `Scenario` 字段（schema.py:8-41）

| 练习记号 | Scenario 字段 | 代码 |
| --- | --- | --- |
| B | `batch` | 每批请求数 |
| P | `tokens` | 本次 forward 每条请求处理的新 token 数 |
| S | `history` | 每条请求**已有**的 KV 上下文长度（decode 场景是 8192，练习场景 1 是 4096） |
| m=BP | `rows`（property） | `batch * tokens`，所有 linear 算子的输入行数（qwen3.py:81） |
| N_pair | `pairs`（property） | `batch * (tokens*history + tokens*(tokens+1)//2)` |
| m_out | `head_rows`（property） | `output_head="last"` 时 = `batch`（每条只投影最后一个 token） |

`pairs` 是本练习的灵魂（schema.py:31-33）：

```python
pairs = batch * (tokens * history + tokens * (tokens + 1) // 2)
#        └ B ┘    └─ 矩形：新 token × 已有历史 ─┘  └── 三角形：新 token 之间的因果对（含自身）──┘
```

- **矩形项 P·S**：每个新 token 都要对全部 S 个历史位置做 QK 和 AV；
- **三角形项 P(P+1)/2**：新 token 彼此之间的因果配对（第 i 个新 token 看到 0..i，含自身）。

### 3.2 表 2-1 ↔ `build_operators()`（qwen3.py:90-177）

| 表 2-1 行 | 算子 | matrix_flops（单次出现） |
| --- | --- | --- |
| 词嵌入查行 | `embedding`（:90-95） | **0**（只计 `weight_read_bytes = m·h·wb` 的行读取） |
| Q 投影 | `linear("q_proj", m, h, q_width)` | 2·m·4096·4096 |
| K、V 投影 | `linear("k_proj"/"v_proj", m, h, kv_width)` | 各 2·m·4096·1024（GQA：8 KV 头×128=1024） |
| QK^T 与 AV | `qk`、`pv`（:124-149） | 各 **2·valid_scores·d**，valid_scores = 32·pairs ⇒ 合计 4·32·128·N_pair |
| 输出投影 | `linear("o_proj", m, q_width, h)` | 2·m·4096·4096 |
| SwiGLU | `gate/up/down_proj`（:160-169） | 3×2·m·4096·12288 = 6·m·h·f |
| 词表头 | `linear("lm_head", head_rows, h, vocab)`（:174-175） | 2·m_out·4096·151936 |

投影单层合计 = 2q+2o+2k+2v+6·FFN = 3,161,095,929,856 / m flops，除以 m 正是正文每层系数 **385,875,968**（= 2·4096²×2 + 2·4096·1024×2 + 6·4096·12288）；注意力每层系数 4·32·128 = **16,384**。`linear()` 的 FLOPs 约定是 `2·rows·inputs·outputs`（schema.py:103，FMA=2）。

### 3.3 汇总 ↔ 正文公式（summarize，qwen3.py:200-238）

- `backbone_projection_ffn_flops`（:200）：`category=="linear"` 且非 `lm_head` 的 `flops×repeats` 之和 = 36·385,875,968·m；
- `causal_attention_matrix_flops`（:221）：`4·qh·pairs·d·layers` = 36·16,384·N_pair；
- `matrix_flops`（:223）：全部算子合计（含 lm_head）。
- 另有 `rectangular_attention_matrix_flops`（:222）：不做因果掩码的矩形上界 = `rows·(S+P)`——注意 `qk` 算子的 `activation_write_bytes` 按**矩形**分数张量计（FP32 物化），而 FLOPs 只按有效因果对计，两者口径刻意分开（README"限制"第 1 条）。

## 4. 练习解答（全部数值已逐位验证）

### 场景 1：B=4, S=4096, P=1024（m=4096，N_pair=4×(1024×4096+1024×1025/2)=18,876,416）

| 项 | FLOPs | 数值 |
| --- | ---: | ---: |
| 投影与 FFN：36×385,875,968×4096 | 56,899,726,737,408 | **56.900 TFLOPs** |
| 有效因果注意力：36×16,384×18,876,416 | 11,133,763,190,784 | **11.134 TFLOPs** |
| 词表头：2×4×4096×151,936 | 4,978,638,848 | **4.979 GFLOPs** |
| 合计 | 68,038,468,567,040 | **68.038 TFLOPs** |

### 场景 2：B=1, S=4096, P=4096（BP 仍 4096）——先预测再验证

预测逻辑（可直接从代码读出）：

1. **投影与 FFN 相同**：只依赖 `rows=B·P=4096`，与 B、P 的拆分无关（qwen3.py:81 每个 linear 都用 `scenario.rows`）；
2. **注意力变大**：`pairs` 的矩形项 B·P·S 两场景相同（4×1024×4096 = 1×4096×4096 = 16,777,216），差异**全部来自三角形项**——1×(4096×4097/2)=8,390,656 vs 4×(1024×1025/2)=2,099,200，大 4 倍（三角形按 P² 增长，拆成 4 条短序列就把三角形切小了）；
3. **词表头变小**：`head_rows="last" 口径 = batch`，B 从 4 降到 1 ⇒ A 是 B 的 4 倍。

验证：

| 项 | 场景 1 | 场景 2 | 差异 |
| --- | ---: | ---: | --- |
| 投影与 FFN | 56.900 TFLOPs | 56.900 TFLOPs | **相同**（输入行数相同） |
| 有效因果注意力 | 11.134 TFLOPs（N_pair=18,876,416） | 14.845 TFLOPs（N_pair=25,167,872） | 场景 2 大 **33.3%** |
| 词表头 | 4.979 GFLOPs（m_out=4） | 1.245 GFLOPs（m_out=1） | 场景 1 是场景 2 的 4 倍 |
| 合计 | 68.038 TFLOPs | 71.746 TFLOPs | 场景 2 大 5.45% |

**这就是练习标题的答案**：输入行数相同，投影/FFN 的账就相同；注意力账单看的是**查询—键配对数** N_pair，它由"新 token 数"和"能看见的历史长度"共同决定——同样 4096 行输入，一条 4096 长序列整段 prefill 的因果三角形（P²/2 量级）远大于 4 条 1024 短序列之和。

### 场景 3：8K 输入完整处理 vs 复用 6144 前缀补 2048

| 方式 | Scenario | 投影+FFN | 因果注意力 | 词表头 | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| 完整处理 | (B=1, S=0, P=8192) | 113.799 TF | 19.794 TF | 1.245 GF | **133.594 TF** |
| 复用前缀 | (B=1, S=6144, P=2048) | 28.450 TF | 8.659 TF | 1.245 GF | **37.110 TF** |

**节省比例 = 1 − 37.110/133.594 = 72.22%**（精确：1 − 37,110,366,076,928 / 133,594,323,353,600）。

为什么不是省 75%（token 只少了 3/4）？分项比例说明一切：

- 投影+FFN 严格按新 token 数线性缩：2048/8192 = **25.0%**（缓存省掉的正是旧 token 的投影与 FFN 重算——正文 2.1.4 末尾的原话）；
- 注意力只剩 **43.75%**：N_pair 比 = (2048×6144+2048×2049/2) / (8192×8193/2) = 14,681,088/33,558,528 = 0.4375。新 token 仍要对全部 8192 个位置做 QK/AV，省不掉；省掉的只是旧前缀之间那部分因果三角形；
- 词表头 **0%**：两种方式都只投影最后 1 个 token。

对照实验 README 的判断 3："命中前缀省的是计算，不是容量"——结束时 KV 常驻仍是 1.125 GiB。

## 5. 实验脚本 `run.py` 解析（142 行）

1. `SCENARIOS`（:23-28）：四个已复算结果文件名 → 场景标签；
2. `load()`（:47）：读文件、缺文件时提示先 reproduce；
3. `group_rows()`（:54）：把 23 个算子按 `GROUP`（:30-41）归成 14 个功能组，每组 FLOPs/权重读/激活读写按 `repeats` 累加，按 FLOPs 降序——README 判断 4"prefill 的工作集中在 SwiGLU（89.1 of 133.6 TFLOPs）"就来自这张表；
4. 三项账（:82-93）：`capacity`（常驻权重+KV）、`reading`（权重读+历史读）、`writing`（新写 KV+激活写）各自独立提取——对应"不能合成一项显存需求"的实验要求；
5. 输出 Markdown 表（:109-134）。

## 6. 建议阅读顺序与动手验证

阅读顺序：

1. `manuscripts/02-模型架构.md` 表 2-1 及其后汇总公式（本练习的一切数值来源）；
2. `calculations/src/infra_calc/schema.py`（136 行，`Scenario.pairs` 是核心）；
3. `calculations/src/infra_calc/models/qwen3.py` 的 `build_operators`/`summarize`；
4. `experiments/ch02/02-02/run.py` 与 `results/qwen3-8b.md`；
5. `calculations/src/infra_calc/models/__init__.py`（分发逻辑，dense/MoE/LLaMA 各走各的 adapter）。

动手验证：

```bash
cd ai-infra-book
# 场景 1（JSON 里搜 backbone/causal_attention/matrix_flops 三个 summary 字段）
python3 calculations/calc.py forward --model qwen3-8b --batch 4 --history 4096 --tokens 1024 | python3 -c "
import json,sys; d=json.load(sys.stdin); s=d['summary']
print('投影+FFN:', s['backbone_projection_ffn_flops'])
print('因果注意力:', s['causal_attention_matrix_flops'])
print('合计:', s['matrix_flops'])"
# 场景 2：--batch 1 --history 4096 --tokens 4096
# 场景 3 已有现成文件：calculations/results/qwen3-8b-{prefill-8192,prefix-6144-plus-2048}.json
```

进阶自问：

- `--output-head all` 时场景 1/2 的词表头各是多少？（m_out=m=4096，两场景**相同**：2×4096×4096×151936 = 5.10 TFLOPs——"last 口径"正是场景 1 词表头大的原因。）
- 让场景 2 的注意力降到与场景 1 相同需要 S 多大？（保持 P=4096、B=1，解 P·S+P(P+1)/2 = 18,876,416 ⇒ 4096·S = 10,485,760 ⇒ S=2560——三角形项 P(P+1)/2=8,390,656 固定不变，只能压缩矩形项；反向问题"B=4、P=1024 时 S 多大追平场景 2"则解出 S≈5787，可体会拆批与历史长度对注意力的双向影响。）
- batch=64 decode 的 `causal_attention_matrix_flops` = 309,275,394,048，用 pairs 公式手算验证（pairs = 64×(1×8192+1) = 524,352；16,384×524,352×36 = ✓）。
