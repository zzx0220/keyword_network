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

## 安装

```bash
pip install pymupdf scikit-learn sentence-transformers networkx matplotlib pandas numpy
```

默认结果保存在 `keyword_network_results/`，包括关键词频率、论文—关键词矩阵、共现矩阵、GraphML、PNG 和可交互 HTML 网络。
