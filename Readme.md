# 文献关键词共现网络

从一组英文 PDF 中自动提取关键词，或统计用户指定的关键词，然后生成论文级共现网络及 CSV 结果。

## 自定义关键词

创建一个 UTF-8 编码的文本文件，关键词之间使用英文逗号 `,` 分隔。关键词本身可以包含空格，例如：

```text
climate change, renewable energy, carbon emissions, biodiversity
```

运行：

```bash
python literature_keyword_network.py ./papers --keywords-file keywords.txt
```

默认的 `guided` 模式会把这些词作为主题定义，其作用与原 template 中的 specific words 相同：

- 定位每篇论文中的主题相关句子及其上下文；
- 从用户词自动推导词汇锚点，过滤偏离主题的候选词；
- 提高用户词和主题短语的排名；
- 防止两个用户明确给出的概念被语义模型自动合并。

可用 `--context-window` 调整主题句前后保留的句子数：

```bash
python literature_keyword_network.py ./papers --keywords-file keywords.txt --context-window 2
```

如果只想统计输入词本身，使用 `exact` 模式：

```bash
python literature_keyword_network.py ./papers --keywords-file keywords.txt --keyword-mode exact
```

`exact` 会在 PDF 全文中进行不区分大小写的完整词/词组匹配，不执行 TF-IDF 筛选或语义合并；未命中的词也会以频次 0 保留在 CSV 中。两种模式都会忽略输入文件中的空项、清理首尾空格，并按首次出现顺序去重。

`--custom-keywords` 是 `--keywords-file` 的等价别名。

## 自动提取

```bash
python literature_keyword_network.py ./papers
python literature_keyword_network.py ./papers --top-n 35 --min-df 1
```

自动提取模式会分析 PDF 全文，不依赖任何预设学科词表。它使用通用 TF-IDF 选择 1–3 词短语，再用 `sentence-transformers` 保守合并语义相似的候选词；首次运行可能需要下载模型。

## 语义相似节点分组

程序会在自动合并之外，进一步将仍然保留但语义相近的节点标记为 semantic group。PNG 和交互式 HTML 会使用同色半透明圆角框包围同组节点，GraphML 节点中会写入 `semantic_group` 属性，并额外生成 `semantic_groups.csv` 供审计。

默认分组阈值为 `0.72`，越高越严格：

```bash
python literature_keyword_network.py ./papers --keywords-file keywords.txt --group-threshold 0.78
```

设为 `0` 可关闭语义框选：

```bash
python literature_keyword_network.py ./papers --keywords-file keywords.txt --group-threshold 0
```

默认情况下，相似度达到 `--merge-threshold` 的高度同义词仍会先合并为一个节点。如果希望尽量保留每个原始节点，再通过框选表示同义关系，可使用：

```bash
python literature_keyword_network.py ./papers --keywords-file keywords.txt \
  --merge-threshold 1.0 --group-threshold 0.72
```

## 边宽频率标准化

交互式 `keyword_network.html` 的顶部工具栏包含 **Normalize edge width** 选项：

- 未勾选时，边宽按两个关键词共同出现的论文数显示。
- 勾选后，边宽按 `cooccurrence / sqrt(frequency1 × frequency2)` 显示。

该归一化值为 0–1，可减少高频词因文章集合主题偏差而产生的边宽优势。切换只改变边的视觉粗细，不会改变节点、边或 CSV 数据。顶部的 Association 滑块仍可独立用于过滤较弱连接。

## 安装

```bash
pip install pymupdf scikit-learn sentence-transformers networkx matplotlib pandas numpy
```

默认结果保存在 `keyword_network_results/`，包括关键词频率、论文—关键词矩阵、共现矩阵、GraphML、PNG 和可交互 HTML 网络。
