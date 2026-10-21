# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库是《深入理解 AI Infra：量化分析与系统设计》（李博杰 著，[bojieli/ai-infra-book](https://github.com/bojieli/ai-infra-book)）的**个人学习工程**，用于记录学习过程中的提问、答疑与笔记。本仓库自身不含书稿内容。

## 目录结构

- `ai-infra-book/` — 从 GitHub 克隆并精简的上游书仓（已删除多语言版本、PDF 生成、Web 发布、存档等冗余部分，只保留学习所需内容）。**仅作只读参考资料，不要修改。**
- `docs/` — 学习答疑文档，所有问答记录保存在这里。

## 语言约定

一切问答讨论和文档记录必须使用中文，包括写入 `docs/` 的答疑文档和本仓库产生的所有 Markdown 文件。

## CodeGraph 符号索引

`ai-infra-book/` 已用 CodeGraph 建立符号索引（`ai-infra-book/.codegraph/`）。分析书中代码、计算工具或实验时，**优先使用 codegraph 而不是 grep/find**：

- MCP 工具：`codegraph_explore`，必须传 `projectPath`（指向 `ai-infra-book/`，仓库根目录本身没有索引，无默认项目）。
- Shell 兜底：在 `ai-infra-book/` 下执行 `codegraph explore "<符号名或问题>"`。

## docs/ 答疑文档规范

- 命名：`<序号>-ch<章节号>-<问题关键字>.md`，如 `00-ch01-显存估算.md`。序号为两位递增数字，章节号与问题所属章节对应（ch01–ch12）。
- 保存新问题前，先查看 `docs/` 已有文档：主题相关的补充到已有文档；不相关的新建文档，序号取现有最大序号加一。
- 文档内容用中文，记录问题、推导/分析过程和结论。

## ai-infra-book 结构速览

| 目录 | 内容 |
| --- | --- |
| `manuscripts/` | 正文章节 `00-前言.md` 至 `12-端边云协同.md`（事实与推导的最终来源） |
| `calculations/` | 全书量化计算工具，入口 `calc.py` CLI；Python 3.10+ 标准库即可运行，无需 GPU 或模型权重 |
| `experiments/` | 配套实验，按 `chXX/XX-YY/` 组织，每项含 `run.py`（独立入口）和 `README.md`（题目、固定输入、结果） |
| `skills/` | 12 个按章节组织的量化分析方法指南（各含 `SKILL.md`），可按任务加载 |
| `research/`、`case-studies/` | 作者写作复核记录与案例分析 |

书中正文的算例推导以 `manuscripts/` 为准；需要复算或代入新模型/硬件时，用 `calculations/` 的 CLI，不要把书中算例结果当作当前硬件基准。

## 常用命令（在 `ai-infra-book/` 目录下执行）

```bash
# 量化计算工具
python3 calculations/calc.py models                                        # 查看模型支持情况
python3 calculations/calc.py forward --model qwen3-8b --tokens 8192 --format md
python3 calculations/calc.py reproduce                                     # 复算已实现结果

# 计算工具单元测试
python3 -m unittest discover -s calculations/tests -v

# 运行单个实验（任取 experiments/chXX/ 下一项）
cd experiments/ch04/04-05 && python3 run.py
```
