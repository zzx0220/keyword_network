#!/usr/bin/env python3
"""
从多篇英文 PDF 中提取关键词，保守合并近义词，并绘制关键词共现网络。

网络含义：
  - 节点大小：包含该关键词/概念的论文数量（document frequency）
  - 边的粗细：两个关键词/概念共同出现的论文数量

默认输出到 ./keyword_network_results：
  paper_summary.csv              每篇论文的处理概况
  keyword_candidates.csv         通过通用筛选的候选关键词
  excluded_keyword_candidates.csv 被通用筛选排除的高分词（供审计）
  keyword_clusters.csv           原词到合并后概念的映射及相似度
  merge_suggestions.csv          未自动合并、但值得人工检查的相似词对
  semantic_groups.csv            网络节点的语义相似分组
  keyword_frequency.csv          合并后的关键词文献频率
  paper_keyword_matrix.csv       论文 × 关键词二值矩阵
  keyword_cooccurrence.csv       关键词 × 关键词共现矩阵
  strongest_connections.csv      最强连接表
  keyword_network.png/.html/.graphml

安装：
  pip install pymupdf scikit-learn sentence-transformers networkx \
      matplotlib pandas numpy

运行：
  python literature_keyword_network.py ./papers
  python literature_keyword_network.py ./papers --top-n 35 --min-df 1
  python literature_keyword_network.py ./papers --keywords-file keywords.txt

首次运行 sentence-transformers 时需要联网下载 all-MiniLM-L6-v2 模型。
如果扫描版 PDF 无可提取文字，请先对其进行 OCR。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 仅排除通用学术写作套语；不在核心程序中预设任何学科词表。
CUSTOM_STOPWORDS = {
    "study", "studies", "result", "results", "method", "methods",
    "experiment", "experiments", "participant", "participants", "subject",
    "subjects", "condition", "conditions", "trial", "trials", "figure",
    "figures", "table", "tables", "using", "used", "use", "different",
    "significant", "significantly", "however", "therefore",
    "respectively", "paper", "article", "author", "authors", "copyright",
}


@dataclass
class Paper:
    name: str
    path: Path
    pages: int
    text: str


@dataclass(frozen=True)
class TopicConfig:
    """由用户关键词动态生成的主题约束，取代旧版硬编码领域词表。"""
    seeds: frozenset[str]
    anchor_words: frozenset[str]
    anchor_prefixes: tuple[str, ...]
    focused_single_terms: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取 PDF 关键词、合并近义词并绘制论文级共现网络。"
    )
    parser.add_argument("pdf_folder", type=Path, help="存放 PDF 的文件夹")
    parser.add_argument("--output", type=Path, default=Path("keyword_network_results"))
    parser.add_argument(
        "--keywords-file", "--custom-keywords", dest="keywords_file", type=Path,
        help="自定义关键词文本文件；关键词用英文逗号分隔，可包含空格",
    )
    parser.add_argument(
        "--keyword-mode", choices=("guided", "exact"), default="guided",
        help="guided 用输入词引导自动发现；exact 仅统计输入词",
    )
    parser.add_argument("--top-n", type=int, default=30, help="合并前保留的候选词数")
    parser.add_argument("--min-df", type=int, default=1,
                        help="候选词至少出现于多少篇论文；约 10 篇时建议 1")
    parser.add_argument("--min-edge", type=int, default=2,
                        help="边至少共同出现于多少篇论文")
    parser.add_argument("--context-window", type=int, default=1,
                        help="guided 模式中，命中主题词的句子前后各保留几句")
    parser.add_argument("--merge-threshold", type=float, default=0.88,
                        help="自动语义合并阈值；越高越保守")
    parser.add_argument("--suggest-threshold", type=float, default=0.78,
                        help="人工复核候选对的最低相似度")
    parser.add_argument("--group-threshold", type=float, default=0.72,
                        help="将语义相似节点框在一起的最低相似度；0 关闭")
    parser.add_argument("--max-df", type=float, default=1.0,
                        help="忽略出现比例高于此值的候选词")
    parser.add_argument("--min-association", type=float, default=0.35,
                        help="绘图边的最低归一化关联强度（0–1）")
    parser.add_argument("--max-edges-per-node", type=int, default=4,
                        help="绘图中每个节点最多保留几条最强边；0 表示不限制")
    parser.add_argument("--seed", type=int, default=42, help="网络布局随机种子")
    parser.add_argument("--show", action="store_true", help="保存后显示网络图")
    return parser.parse_args()


def normalize_term(term: str) -> str:
    term = term.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", term).strip(" .,:;()[]{}")


def normalize_custom_keyword(term: str) -> str:
    """清理用户输入，但不应用自动提取模式的同义词合并规则。"""
    term = term.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", term).strip(" .,:;()[]{}\"'")


def load_custom_keywords(path: Path) -> list[str]:
    """读取逗号分隔的 UTF-8 关键词，按首次出现顺序去重。"""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"关键词文件必须使用 UTF-8 编码：{path}") from exc

    keywords: list[str] = []
    seen: set[str] = set()
    for raw in content.split(","):
        keyword = normalize_custom_keyword(raw)
        if keyword and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)
    if not keywords:
        raise ValueError(f"关键词文件中没有可用内容：{path}")
    return keywords


def build_topic_config(keywords: list[str]) -> TopicConfig:
    """从用户词表推导主题锚点和单词白名单。"""
    seeds = frozenset(normalize_term(keyword) for keyword in keywords)
    tokenized = [
        [word for word in seed.split() if word not in ENGLISH_STOP_WORDS]
        for seed in seeds
    ]
    anchor_words = frozenset(word for words in tokenized for word in words)
    # 五字符前缀可覆盖常见词形变化，同时比过短词干更不容易误命中。
    anchor_prefixes = tuple(sorted({word[:5] for word in anchor_words if len(word) >= 5}))
    focused_single_terms = frozenset(
        seed for seed in seeds if len(seed.split()) == 1
    ) | frozenset(words[-1] for words in tokenized if words)
    return TopicConfig(seeds, anchor_words, anchor_prefixes, focused_single_terms)


def extract_pdf_text(path: Path) -> tuple[str, int]:
    chunks: list[str] = []
    try:
        with fitz.open(path) as doc:
            pages = len(doc)
            for page in doc:
                chunks.append(page.get_text("text") or "")
        return "\n".join(chunks), pages
    except Exception as exc:
        print(f"[跳过] 无法读取 {path.name}: {exc}", file=sys.stderr)
        return "", 0


def clean_text(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    protected = re.sub(
        r"\b(et al|e\.g|i\.e|Fig|Dr)\.",
        lambda match: match.group(0)[:-1] + "<DOT>",
        text,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    return [part.replace("<DOT>", ".").strip() for part in parts if len(part.strip()) >= 30]


def relevant_context(text: str, topic: TopicConfig, window: int) -> str:
    """保留命中用户主题词的句子及其邻近句，复现原 template 的聚焦机制。"""
    sentences = split_sentences(text)
    exact_patterns = [_custom_keyword_pattern(seed) for seed in topic.seeds]
    # 原 template 显式列举了 prediction/predictive 等变体；现在用动态前缀匹配实现同样的词形覆盖。
    variant_patterns = [
        re.compile(r"(?<![A-Za-z0-9])" + re.escape(prefix) + r"[A-Za-z-]*(?![A-Za-z0-9])",
                   re.IGNORECASE)
        for prefix in topic.anchor_prefixes
    ]
    selected: set[int] = set()
    for index, sentence in enumerate(sentences):
        if any(pattern.search(sentence) for pattern in exact_patterns + variant_patterns):
            selected.update(range(
                max(0, index - window), min(len(sentences), index + window + 1)
            ))
    return " ".join(sentences[index] for index in sorted(selected))


def load_papers(
    folder: Path, guided_topic: TopicConfig | None = None, context_window: int = 1
) -> tuple[list[Paper], list[dict]]:
    pdfs = sorted(folder.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"在 {folder.resolve()} 中没有找到 PDF。")
    papers: list[Paper] = []
    summary: list[dict] = []
    for path in pdfs:
        raw, pages = extract_pdf_text(path)
        full = clean_text(raw)
        analyzed = (
            relevant_context(full, guided_topic, context_window)
            if full and guided_topic else full
        )
        status = "included" if len(analyzed) >= 100 else "excluded_too_little_text"
        summary.append({
            "paper": path.stem, "file": str(path), "pages": pages,
            "full_text_characters": len(full),
            "analyzed_text_characters": len(analyzed), "status": status,
        })
        if status == "included":
            papers.append(Paper(path.stem, path, pages, analyzed))
            label = "主题相关文本" if guided_topic else "可用全文"
            print(f"[纳入] {path.name}: {label} {len(analyzed):,} 字符")
        else:
            reason = "主题相关文本太少" if guided_topic else "可用文本太少"
            print(f"[跳过] {path.name}: {reason}（可能需要 OCR 或调整关键词）")
    return papers, summary


def _custom_keyword_pattern(keyword: str) -> re.Pattern[str]:
    # 词之间允许 PDF 提取后出现任意空白；边界判断避免命中更长单词。
    body = r"\s+".join(re.escape(part) for part in keyword.split())
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.IGNORECASE)


def extract_custom_keywords(
    documents: list[str], keywords: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """直接匹配用户指定的关键词，不排名、过滤或自动合并。"""
    terms = np.array(keywords, dtype=object)
    patterns = [_custom_keyword_pattern(keyword) for keyword in keywords]
    binary = np.array(
        [[bool(pattern.search(document)) for pattern in patterns] for document in documents],
        dtype=np.uint8,
    )
    df = binary.sum(axis=0).astype(int)
    # 自定义模式没有 TF-IDF；保留同名列便于下游兼容和审计。
    mean_tfidf = np.zeros(len(keywords), dtype=float)
    candidates = pd.DataFrame({
        "keyword": terms,
        "document_frequency": df,
        "mean_tfidf": mean_tfidf,
        "ranking_score": mean_tfidf,
    })
    excluded = pd.DataFrame(columns=[
        "keyword", "document_frequency", "mean_tfidf", "ranking_score", "exclusion_reason"
    ])
    return terms, binary, df, mean_tfidf, candidates, excluded


def term_filter_reason(term: str, topic: TopicConfig | None = None) -> str | None:
    words = term.split()
    if not 1 <= len(words) <= 3 or re.search(r"\d", term):
        return "invalid_length_or_number"
    if any(len(w.strip("-")) < 3 for w in words):
        return "short_token"
    if len(words) == 1:
        if words[0] in ENGLISH_STOP_WORDS or words[0] in CUSTOM_STOPWORDS:
            return "general_stopword"
        if topic is not None and term not in topic.focused_single_terms:
            return "non_topic_single_word"
    else:
        if all(w in ENGLISH_STOP_WORDS or w in CUSTOM_STOPWORDS for w in words):
            return "general_phrase"
        if topic is not None and not any(
            word in topic.anchor_words
            or any(word.startswith(prefix) for prefix in topic.anchor_prefixes)
            for word in words
        ):
            return "no_topic_anchor"
    return None


def extract_candidates(
    documents: list[str], top_n: int, min_df: int, max_df: float,
    topic: TopicConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 3), min_df=min_df,
        max_df=max_df, max_features=12000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b", sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        raise RuntimeError("无法提取关键词；可尝试 --min-df 1 或增加 PDF。") from exc
    terms = np.array([normalize_term(t) for t in vectorizer.get_feature_names_out()])
    mean_tfidf = np.asarray(matrix.mean(axis=0)).ravel()
    binary = (matrix > 0).astype(np.uint8).toarray()
    df = binary.sum(axis=0)

    reasons = [term_filter_reason(str(term), topic) for term in terms]
    valid = [i for i, reason in enumerate(reasons) if reason is None]
    # guided 模式优先用户给出的完整词组，再优先其他主题相关短语。
    phrase_boost = np.array([
        3.0 if topic is not None and term in topic.seeds and len(term.split()) > 1
        else 1.8 if len(term.split()) > 1
        else 0.75 if topic is not None
        else 1.0
        for term in terms
    ])
    scores = mean_tfidf * (1.0 + np.log1p(df)) * phrase_boost
    ranked = sorted(valid, key=lambda i: (scores[i], df[i]), reverse=True)

    # normalize 后可能产生同名候选；保留分数最高的一列。
    selected: list[int] = []
    seen: set[str] = set()
    for i in ranked:
        if terms[i] not in seen:
            selected.append(i)
            seen.add(terms[i])
        if len(selected) >= top_n:
            break
    if not selected:
        raise RuntimeError("过滤后没有候选关键词；请增加论文或调整参数。")

    selected_terms = terms[selected]
    selected_binary = binary[:, selected]
    candidate_df = pd.DataFrame({
        "keyword": selected_terms,
        "document_frequency": df[selected].astype(int),
        "mean_tfidf": mean_tfidf[selected],
        "ranking_score": scores[selected],
    }).sort_values(["ranking_score", "document_frequency"], ascending=False)
    excluded_ranked = sorted(
        (i for i, reason in enumerate(reasons) if reason is not None),
        key=lambda i: scores[i], reverse=True,
    )[:max(100, top_n * 3)]
    excluded_df = pd.DataFrame({
        "keyword": terms[excluded_ranked],
        "document_frequency": df[excluded_ranked].astype(int),
        "mean_tfidf": mean_tfidf[excluded_ranked],
        "ranking_score": scores[excluded_ranked],
        "exclusion_reason": [reasons[i] for i in excluded_ranked],
    })
    return (selected_terms, selected_binary, df[selected], mean_tfidf[selected],
            candidate_df, excluded_df)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def semantic_similarity(terms: list[str] | np.ndarray) -> np.ndarray:
    """使用通用句子 embedding 计算关键词两两语义相似度。"""
    # 某些 macOS/Conda 组合在 PyTorch 多 OpenMP 线程编码短文本时会崩溃。
    # 关键词量很小，单线程更稳定，对实际速度影响可忽略。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 sentence-transformers。请运行：pip install sentence-transformers"
        ) from exc

    print("正在计算关键词语义相似度……")
    # CPU 对 macOS/Windows/Linux 都更稳定，且关键词数量较小，无需启动 GPU/MPS。
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    values = [str(term) for term in terms]
    embeddings = model.encode(values, normalize_embeddings=True, show_progress_bar=False)
    return cosine_similarity(embeddings)


def semantic_groups(
    terms: list[str], similarity: np.ndarray, threshold: float
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    用 complete-link 组成视觉语义组，防止 A≈B、B≈C 把差异较大的 A/C 链式并入。
    只输出至少包含两个节点的组。
    """
    count = len(terms)
    if threshold <= 0:
        columns = [
            "semantic_group", "keyword", "group_size",
            "minimum_similarity_to_group", "mean_similarity_to_group",
        ]
        return {}, pd.DataFrame(columns=columns)
    if similarity.shape != (count, count):
        raise ValueError("语义相似度矩阵与关键词数量不匹配。")
    uf = UnionFind(count)

    def members(root: int) -> list[int]:
        return [index for index in range(count) if uf.find(index) == root]

    pairs = sorted(
        ((float(similarity[i, j]), i, j) for i in range(count) for j in range(i + 1, count)),
        reverse=True,
    )
    for score, left_index, right_index in pairs:
        if score < threshold:
            break
        left_root, right_root = uf.find(left_index), uf.find(right_index)
        if left_root == right_root:
            continue
        left, right = members(left_root), members(right_root)
        if min(float(similarity[a, b]) for a in left for b in right) >= threshold:
            uf.union(left_root, right_root)

    raw_groups: dict[int, list[int]] = {}
    for index in range(count):
        raw_groups.setdefault(uf.find(index), []).append(index)
    grouped = sorted(
        (indices for indices in raw_groups.values() if len(indices) >= 2),
        key=lambda indices: min(indices),
    )

    assignments: dict[str, str] = {}
    rows: list[dict] = []
    for number, indices in enumerate(grouped, start=1):
        group_id = f"G{number}"
        for index in indices:
            peers = [other for other in indices if other != index]
            assignments[terms[index]] = group_id
            rows.append({
                "semantic_group": group_id,
                "keyword": terms[index],
                "group_size": len(indices),
                "minimum_similarity_to_group": round(
                    min(float(similarity[index, other]) for other in peers), 4
                ),
                "mean_similarity_to_group": round(
                    float(np.mean([similarity[index, other] for other in peers])), 4
                ),
            })
    columns = [
        "semantic_group", "keyword", "group_size",
        "minimum_similarity_to_group", "mean_similarity_to_group",
    ]
    return assignments, pd.DataFrame(rows, columns=columns)


def semantic_clusters(
    terms: np.ndarray,
    doc_frequency: np.ndarray,
    merge_threshold: float,
    suggest_threshold: float,
    topic: TopicConfig | None = None,
) -> tuple[list[list[int]], np.ndarray, pd.DataFrame]:
    sim = semantic_similarity(terms)
    pairs = sorted(
        ((float(sim[i, j]), i, j) for i in range(len(terms)) for j in range(i + 1, len(terms))),
        reverse=True,
    )
    uf = UnionFind(len(terms))

    def protected_pair(left: int, right: int) -> bool:
        return (
            topic is not None
            and str(terms[left]) in topic.seeds
            and str(terms[right]) in topic.seeds
        )

    # 只有当两个簇中所有跨簇词对均超过阈值时才合并。
    # 这比简单的贪心单链接更保守，可避免 A≈B、B≈C 导致 A 与 C 被链式误合并。
    def members(root: int) -> list[int]:
        return [i for i in range(len(terms)) if uf.find(i) == root]

    for similarity, i, j in pairs:
        if similarity < merge_threshold:
            break
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        left, right = members(ri), members(rj)
        if any(protected_pair(a, b) for a in left for b in right):
            continue
        if min(float(sim[a, b]) for a in left for b in right) >= merge_threshold:
            uf.union(ri, rj)

    groups: dict[int, list[int]] = {}
    for i in range(len(terms)):
        groups.setdefault(uf.find(i), []).append(i)
    clusters = list(groups.values())

    suggestions = []
    for similarity, i, j in pairs:
        if similarity < suggest_threshold:
            break
        if uf.find(i) == uf.find(j):
            continue
        suggestions.append({
            "keyword_1": terms[i], "keyword_2": terms[j],
            "cosine_similarity": round(similarity, 4),
            "protected_from_merge": protected_pair(i, j),
        })
    return clusters, sim, pd.DataFrame(suggestions)


def merge_keyword_matrix(
    terms: np.ndarray,
    binary: np.ndarray,
    doc_frequency: np.ndarray,
    mean_tfidf: np.ndarray,
    clusters: list[list[int]],
    similarity: np.ndarray,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    merged_terms: list[str] = []
    columns: list[np.ndarray] = []
    audit_rows: list[dict] = []

    for cluster in clusters:
        # 代表词优先选择跨论文覆盖较高、随后 TF-IDF 较高、最后更简短者。
        rep_idx = max(cluster, key=lambda i: (int(doc_frequency[i]), float(mean_tfidf[i]), -len(terms[i])))
        representative = str(terms[rep_idx])
        merged_terms.append(representative)
        columns.append(binary[:, cluster].max(axis=1))
        for i in cluster:
            audit_rows.append({
                "original_keyword": terms[i],
                "merged_keyword": representative,
                "similarity_to_representative": round(float(similarity[i, rep_idx]), 4),
                "automatically_merged": i != rep_idx,
                "original_document_frequency": int(doc_frequency[i]),
            })

    merged_binary = np.column_stack(columns).astype(np.uint8)
    order = sorted(
        range(len(merged_terms)),
        key=lambda i: (int(merged_binary[:, i].sum()), merged_terms[i]), reverse=True,
    )
    merged_terms = [merged_terms[i] for i in order]
    merged_binary = merged_binary[:, order]
    audit = pd.DataFrame(audit_rows).sort_values(["merged_keyword", "original_keyword"])
    return merged_terms, merged_binary, audit


def build_network(
    terms: list[str], binary: np.ndarray, min_edge: int,
    min_association: float, max_edges_per_node: int,
    group_assignments: dict[str, str] | None = None,
) -> tuple[nx.Graph, np.ndarray]:
    count_matrix = binary.astype(np.int64, copy=False)
    cooccurrence = count_matrix.T @ count_matrix
    graph = nx.Graph()
    for i, term in enumerate(terms):
        graph.add_node(
            term,
            frequency=int(binary[:, i].sum()),
            semantic_group=(group_assignments or {}).get(term, ""),
        )

    candidates: list[tuple[float, int, int, int]] = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            weight = int(cooccurrence[i, j])
            denominator = np.sqrt(float(cooccurrence[i, i]) * float(cooccurrence[j, j]))
            association = weight / denominator if denominator else 0.0
            if weight >= min_edge and association >= min_association:
                candidates.append((association, weight, i, j))

    # 从最强边开始加入，并限制节点度数；原始共现矩阵仍完整写入 CSV。
    degrees = [0] * len(terms)
    for association, weight, i, j in sorted(candidates, reverse=True):
        if max_edges_per_node > 0 and (
            degrees[i] >= max_edges_per_node or degrees[j] >= max_edges_per_node
        ):
            continue
        graph.add_edge(
            terms[i], terms[j], weight=weight,
            association_strength=round(float(association), 6),
        )
        degrees[i] += 1
        degrees[j] += 1
    return graph, cooccurrence


GROUP_COLORS = (
    "#F59E0B", "#10B981", "#8B5CF6", "#EF4444", "#06B6D4",
    "#EC4899", "#84CC16", "#F97316", "#6366F1", "#14B8A6",
)


def _visible_grouped_graph(graph: nx.Graph) -> nx.Graph:
    drawn = graph.copy()
    hidden = [
        node for node in nx.isolates(drawn)
        if not drawn.nodes[node].get("semantic_group")
    ]
    drawn.remove_nodes_from(hidden)
    return drawn


def _grouped_layout(graph: nx.Graph, seed: int, x_scale: float = 1.0,
                    y_scale: float = 1.0) -> dict[str, np.ndarray]:
    """语义组内加入仅用于布局的隐形引力，使分组边界更清晰。"""
    layout_graph = graph.copy()
    for source, target, data in layout_graph.edges(data=True):
        data["layout_weight"] = max(0.05, float(data.get("association_strength", 0.0)))
    groups: dict[str, list[str]] = {}
    for node, data in layout_graph.nodes(data=True):
        if data.get("semantic_group"):
            groups.setdefault(str(data["semantic_group"]), []).append(node)
    for members in groups.values():
        for i, source in enumerate(members):
            for target in members[i + 1:]:
                if layout_graph.has_edge(source, target):
                    layout_graph[source][target]["layout_weight"] += 3.0
                else:
                    layout_graph.add_edge(source, target, layout_weight=3.0)
    positions = nx.spring_layout(
        layout_graph, seed=seed,
        k=2.2 / np.sqrt(max(1, layout_graph.number_of_nodes())),
        iterations=500, weight="layout_weight",
    )
    return {
        node: np.array([float(point[0]) * x_scale, float(point[1]) * y_scale])
        for node, point in positions.items()
    }


def draw_network(graph: nx.Graph, output: Path, seed: int, show: bool) -> None:
    # 普通孤立节点只保留在数据文件中；语义分组节点即使无共现边也会显示。
    drawn = _visible_grouped_graph(graph)
    if drawn.number_of_nodes() == 0:
        print("[提示] 当前 min-edge 下没有可绘制的连接；CSV 结果仍已保存。")
        return

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = _grouped_layout(drawn, seed)
    frequencies = np.array([drawn.nodes[n]["frequency"] for n in drawn.nodes()], dtype=float)
    node_sizes = 450 + 550 * frequencies
    degrees = np.array([drawn.degree(n, weight="weight") for n in drawn.nodes()], dtype=float)
    colors = plt.cm.Blues(0.35 + 0.55 * degrees / max(1.0, degrees.max()))
    max_edge_weight = max((data["weight"] for _, _, data in drawn.edges(data=True)), default=1)
    widths = [0.5 + 4.0 * drawn[u][v]["weight"] / max_edge_weight for u, v in drawn.edges()]

    semantic_sets: dict[str, list[str]] = {}
    for node, data in drawn.nodes(data=True):
        if data.get("semantic_group"):
            semantic_sets.setdefault(str(data["semantic_group"]), []).append(node)
    for color_index, (group_id, members) in enumerate(sorted(semantic_sets.items())):
        points = np.array([pos[node] for node in members])
        padding = 0.09
        left, bottom = points.min(axis=0) - padding
        right, top = points.max(axis=0) + padding
        color = GROUP_COLORS[color_index % len(GROUP_COLORS)]
        box = FancyBboxPatch(
            (left, bottom), max(right - left, 0.18), max(top - bottom, 0.18),
            boxstyle="round,pad=0.025,rounding_size=0.04",
            facecolor=color, edgecolor=color, alpha=0.12, linewidth=1.8, zorder=0,
        )
        ax.add_patch(box)
        ax.text(left + 0.015, top - 0.005, group_id, color=color, fontsize=9,
                fontweight="bold", va="top", zorder=1)

    nx.draw_networkx_edges(drawn, pos, ax=ax, width=widths, alpha=0.38, edge_color="#6F839B")
    node_borders = [
        GROUP_COLORS[(int(str(drawn.nodes[node]["semantic_group"])[1:]) - 1) % len(GROUP_COLORS)]
        if drawn.nodes[node].get("semantic_group") else "white"
        for node in drawn.nodes()
    ]
    nx.draw_networkx_nodes(drawn, pos, ax=ax, node_size=node_sizes, node_color=colors,
                           edgecolors=node_borders, linewidths=2.0, alpha=0.95)
    nx.draw_networkx_labels(drawn, pos, ax=ax, font_size=9, font_color="#172331")
    ax.set_title("Keyword Co-occurrence Network", fontsize=18, pad=18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "keyword_network.png", dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def write_interactive_network(graph: nx.Graph, output: Path, seed: int) -> None:
    """Write a standalone interactive SVG network without extra dependencies."""
    drawn = _visible_grouped_graph(graph)
    if drawn.number_of_nodes() == 0:
        return

    positions = _grouped_layout(drawn, seed, x_scale=420, y_scale=320)
    nodes = [{
        "id": str(node),
        "x": round(float(positions[node][0]), 3),
        "y": round(float(positions[node][1]), 3),
        "frequency": int(drawn.nodes[node]["frequency"]),
        "weighted_degree": round(float(drawn.degree(node, weight="weight")), 3),
        "semantic_group": str(drawn.nodes[node].get("semantic_group", "")),
    } for node in drawn.nodes()]
    edges = [{
        "source": str(source), "target": str(target),
        "weight": int(data["weight"]),
        "association": float(data["association_strength"]),
    } for source, target, data in drawn.edges(data=True)]
    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")

    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Keyword Co-occurrence Network</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172331}*{box-sizing:border-box}
body{margin:0;background:#f6f8fb}header{display:flex;flex-wrap:wrap;align-items:center;gap:14px;
padding:14px 18px;background:#fff;border-bottom:1px solid #dce3ec}h1{margin:0 auto 0 0;font-size:18px}
label{font-size:13px;color:#4b5f74}input[type=search]{width:210px;padding:7px 10px;border:1px solid #bdc9d7;border-radius:7px}
input[type=range],input[type=checkbox]{vertical-align:middle}button{padding:7px 11px;border:1px solid #bdc9d7;border-radius:7px;background:#fff;cursor:pointer}
#network{width:100vw;height:calc(100vh - 66px);background:#fff;cursor:grab}#network.dragging{cursor:grabbing}
.edge{stroke:#71869d;stroke-opacity:.34;vector-effect:non-scaling-stroke}.node circle{stroke:#fff;stroke-width:2;
vector-effect:non-scaling-stroke;cursor:move}.node text{fill:#172331;font-size:12px;text-anchor:middle;pointer-events:none;
paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}.dim{opacity:.08!important}.highlight circle{stroke:#f59e0b;stroke-width:4}
.semantic-box rect{fill-opacity:.10;stroke-width:2;vector-effect:non-scaling-stroke}.semantic-box text{font-size:12px;font-weight:700}
#tooltip{position:fixed;display:none;pointer-events:none;padding:8px 10px;border-radius:7px;background:rgba(23,35,49,.94);
color:#fff;font-size:12px;line-height:1.45;box-shadow:0 4px 18px rgba(0,0,0,.18)}
#help{position:fixed;left:14px;bottom:12px;color:#64748b;font-size:12px;pointer-events:none}
</style></head><body>
<header><h1>Keyword Co-occurrence Network</h1>
<input id="search" type="search" placeholder="Search keywords…" aria-label="Search keywords">
<label>Association ≥ <span id="thresholdValue">0.00</span>
<input id="threshold" type="range" min="0" max="1" step="0.01" value="0"></label>
<label title="Divide co-occurrence by the geometric mean of the two keyword frequencies">
<input id="normalizeEdges" type="checkbox"> Normalize edge width</label>
<button id="reset" type="button">Reset view</button></header>
<svg id="network" viewBox="-520 -390 1040 780"><g id="viewport"><g id="groups"></g><g id="edges"></g><g id="nodes"></g></g></svg>
<div id="tooltip"></div><div id="help">Drag nodes · drag background to pan · scroll to zoom · hover for details</div>
<script>
const data=__NETWORK_DATA__,svg=document.getElementById('network'),viewport=document.getElementById('viewport'),
groupLayer=document.getElementById('groups'),edgeLayer=document.getElementById('edges'),nodeLayer=document.getElementById('nodes'),tooltip=document.getElementById('tooltip');
const nodeById=new Map(data.nodes.map(n=>[n.id,n])),incident=new Map(data.nodes.map(n=>[n.id,new Set()]));
data.edges.forEach(e=>{incident.get(e.source).add(e.target);incident.get(e.target).add(e.source)});
const maxFrequency=Math.max(...data.nodes.map(n=>n.frequency),1),maxDegree=Math.max(...data.nodes.map(n=>n.weighted_degree),1),
maxEdgeWeight=Math.max(...data.edges.map(e=>e.weight),1);
const groupColors=['#F59E0B','#10B981','#8B5CF6','#EF4444','#06B6D4','#EC4899','#84CC16','#F97316','#6366F1','#14B8A6'];
let transform={x:0,y:0,k:1},drag=null;
const radius=n=>9+15*Math.sqrt(n.frequency/maxFrequency);
const color=n=>{const t=n.weighted_degree/maxDegree;return `hsl(${211-12*t} ${48+24*t}% ${72-30*t}%)`};
const semanticGroups=new Map();data.nodes.forEach(n=>{if(n.semantic_group){if(!semanticGroups.has(n.semantic_group))semanticGroups.set(n.semantic_group,[]);semanticGroups.get(n.semantic_group).push(n)}});
Array.from(semanticGroups.entries()).forEach(([id,members],index)=>{const g=document.createElementNS('http://www.w3.org/2000/svg','g');g.classList.add('semantic-box');
const rect=document.createElementNS('http://www.w3.org/2000/svg','rect'),label=document.createElementNS('http://www.w3.org/2000/svg','text');
const c=groupColors[index%groupColors.length];rect.setAttribute('rx','18');rect.setAttribute('fill',c);rect.setAttribute('stroke',c);
label.setAttribute('fill',c);label.textContent=id;g.append(rect,label);groupLayer.appendChild(g);semanticGroups.set(id,{members,g,rect,label})});
function updateGroups(){semanticGroups.forEach(group=>{const left=Math.min(...group.members.map(n=>n.x-Math.max(38,n.id.length*3.2))),right=Math.max(...group.members.map(n=>n.x+Math.max(38,n.id.length*3.2)));
const top=Math.min(...group.members.map(n=>n.y-radius(n)-30)),bottom=Math.max(...group.members.map(n=>n.y+radius(n)+30));group.rect.setAttribute('x',left);group.rect.setAttribute('y',top);
group.rect.setAttribute('width',right-left);group.rect.setAttribute('height',bottom-top);group.label.setAttribute('x',left+12);group.label.setAttribute('y',top+18)})}
const applyTransform=()=>viewport.setAttribute('transform',`translate(${transform.x} ${transform.y}) scale(${transform.k})`);
function updateEdge(e){const a=nodeById.get(e.source),b=nodeById.get(e.target);e.el.setAttribute('x1',a.x);
e.el.setAttribute('y1',a.y);e.el.setAttribute('x2',b.x);e.el.setAttribute('y2',b.y)}
function updateEdgeWidths(){const normalized=document.getElementById('normalizeEdges').checked;
data.edges.forEach(e=>{const strength=normalized?e.association:e.weight/maxEdgeWeight;e.el.style.strokeWidth=`${.7+4.3*strength}px`})}
data.edges.forEach(e=>{const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.classList.add('edge');
const title=document.createElementNS('http://www.w3.org/2000/svg','title');
title.textContent=`${e.source} ↔ ${e.target}\nCo-occurring papers: ${e.weight}\nNormalized association: ${e.association.toFixed(3)}`;
line.appendChild(title);e.el=line;edgeLayer.appendChild(line);updateEdge(e)});
data.nodes.forEach(n=>{const g=document.createElementNS('http://www.w3.org/2000/svg','g');g.classList.add('node');
g.setAttribute('transform',`translate(${n.x} ${n.y})`);const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
circle.setAttribute('r',radius(n));circle.setAttribute('fill',color(n));if(n.semantic_group){const groupNumber=Number(n.semantic_group.slice(1))-1;circle.setAttribute('stroke',groupColors[groupNumber%groupColors.length]);circle.setAttribute('stroke-width','3')}
const label=document.createElementNS('http://www.w3.org/2000/svg','text');
label.setAttribute('y',radius(n)+16);label.textContent=n.id;g.append(circle,label);nodeLayer.appendChild(g);n.el=g;
g.addEventListener('pointerdown',ev=>{ev.stopPropagation();g.setPointerCapture(ev.pointerId);
drag={type:'node',node:n,sx:ev.clientX,sy:ev.clientY,x:n.x,y:n.y}});
g.addEventListener('mouseenter',()=>{tooltip.innerHTML=`<strong>${n.id}</strong><br>Papers: ${n.frequency}<br>Weighted connections: ${n.weighted_degree}${n.semantic_group?`<br>Semantic group: ${n.semantic_group}`:''}`;
tooltip.style.display='block';data.nodes.forEach(o=>o.el.classList.toggle('dim',o.id!==n.id&&!incident.get(n.id).has(o.id)));
data.edges.forEach(e=>e.el.classList.toggle('dim',e.source!==n.id&&e.target!==n.id))});
g.addEventListener('mousemove',ev=>{tooltip.style.left=`${ev.clientX+14}px`;tooltip.style.top=`${ev.clientY+14}px`});
g.addEventListener('mouseleave',()=>{tooltip.style.display='none';data.nodes.forEach(o=>o.el.classList.remove('dim'));
data.edges.forEach(e=>e.el.classList.remove('dim'))})});
svg.addEventListener('pointerdown',ev=>{svg.setPointerCapture(ev.pointerId);svg.classList.add('dragging');
drag={type:'pan',sx:ev.clientX,sy:ev.clientY,x:transform.x,y:transform.y}});
svg.addEventListener('pointermove',ev=>{if(!drag)return;if(drag.type==='pan'){transform.x=drag.x+ev.clientX-drag.sx;
transform.y=drag.y+ev.clientY-drag.sy;applyTransform()}else{drag.node.x=drag.x+(ev.clientX-drag.sx)/transform.k;
drag.node.y=drag.y+(ev.clientY-drag.sy)/transform.k;drag.node.el.setAttribute('transform',`translate(${drag.node.x} ${drag.node.y})`);
data.edges.filter(e=>e.source===drag.node.id||e.target===drag.node.id).forEach(updateEdge);updateGroups()}});
function stopDrag(){drag=null;svg.classList.remove('dragging')}svg.addEventListener('pointerup',stopDrag);svg.addEventListener('pointercancel',stopDrag);
svg.addEventListener('wheel',ev=>{ev.preventDefault();transform.k=Math.min(6,Math.max(.25,transform.k*(ev.deltaY<0?1.12:.89)));applyTransform()},{passive:false});
document.getElementById('threshold').addEventListener('input',ev=>{const value=Number(ev.target.value);
document.getElementById('thresholdValue').textContent=value.toFixed(2);data.edges.forEach(e=>e.el.style.display=e.association>=value?'':'none')});
document.getElementById('normalizeEdges').addEventListener('change',updateEdgeWidths);
document.getElementById('search').addEventListener('input',ev=>{const q=ev.target.value.trim().toLowerCase();
data.nodes.forEach(n=>n.el.classList.toggle('highlight',Boolean(q)&&n.id.toLowerCase().includes(q)))});
document.getElementById('reset').addEventListener('click',()=>{transform={x:0,y:0,k:1};applyTransform()});
updateGroups();updateEdgeWidths();
</script></body></html>
""".replace("__NETWORK_DATA__", payload)
    (output / "keyword_network.html").write_text(page, encoding="utf-8")


def save_results(
    output: Path, papers: list[Paper], summary: list[dict], candidates: pd.DataFrame,
    excluded_candidates: pd.DataFrame,
    terms: list[str], binary: np.ndarray, cooccurrence: np.ndarray,
    audit: pd.DataFrame, suggestions: pd.DataFrame, group_table: pd.DataFrame,
    graph: nx.Graph,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output / "paper_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "keyword_candidates.csv", index=False, encoding="utf-8-sig")
    excluded_candidates.to_csv(
        output / "excluded_keyword_candidates.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(output / "keyword_clusters.csv", index=False, encoding="utf-8-sig")
    suggestions.to_csv(output / "merge_suggestions.csv", index=False, encoding="utf-8-sig")
    group_table.to_csv(output / "semantic_groups.csv", index=False, encoding="utf-8-sig")

    frequency = pd.DataFrame({
        "keyword": terms,
        "number_of_papers": binary.sum(axis=0).astype(int),
        "paper_proportion": np.round(binary.mean(axis=0), 4),
    }).sort_values(["number_of_papers", "keyword"], ascending=[False, True])
    frequency.to_csv(output / "keyword_frequency.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(binary, index=[p.name for p in papers], columns=terms).to_csv(
        output / "paper_keyword_matrix.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(cooccurrence, index=terms, columns=terms).to_csv(
        output / "keyword_cooccurrence.csv", encoding="utf-8-sig"
    )
    edges = sorted(
        ({"keyword_1": u, "keyword_2": v, "cooccurring_papers": d["weight"],
          "association_strength": d["association_strength"]}
         for u, v, d in graph.edges(data=True)),
        key=lambda row: row["cooccurring_papers"], reverse=True,
    )
    pd.DataFrame(edges, columns=["keyword_1", "keyword_2", "cooccurring_papers",
                                 "association_strength"]).to_csv(
        output / "strongest_connections.csv", index=False, encoding="utf-8-sig"
    )
    nx.write_graphml(graph, output / "keyword_network.graphml")


def validate_args(args: argparse.Namespace) -> None:
    if not args.pdf_folder.is_dir():
        raise NotADirectoryError(f"PDF 文件夹不存在：{args.pdf_folder}")
    if args.keywords_file is not None and not args.keywords_file.is_file():
        raise FileNotFoundError(f"关键词文件不存在：{args.keywords_file}")
    if args.keyword_mode == "exact" and args.keywords_file is None:
        raise ValueError("--keyword-mode exact 需要同时提供 --keywords-file。")
    if args.top_n < 2 or args.min_df < 1 or args.min_edge < 1:
        raise ValueError("top-n 至少为 2；min-df 和 min-edge 至少为 1。")
    if not 0 < args.max_df <= 1:
        raise ValueError("max-df 必须在 (0, 1]。")
    if not 0 <= args.suggest_threshold <= args.merge_threshold <= 1:
        raise ValueError("需满足 0 <= suggest-threshold <= merge-threshold <= 1。")
    if not 0 <= args.group_threshold <= args.merge_threshold:
        raise ValueError("需满足 0 <= group-threshold <= merge-threshold。")
    if not 0 <= args.min_association <= 1:
        raise ValueError("min-association 必须在 [0, 1]。")
    if args.max_edges_per_node < 0:
        raise ValueError("max-edges-per-node 不能小于 0。")
    if args.context_window < 0:
        raise ValueError("context-window 不能小于 0。")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        args.output.mkdir(parents=True, exist_ok=True)
        custom_keywords = load_custom_keywords(args.keywords_file) if args.keywords_file else None
        guided = custom_keywords is not None and args.keyword_mode == "guided"
        topic = build_topic_config(custom_keywords) if guided else None
        papers, summary = load_papers(
            args.pdf_folder,
            guided_topic=topic,
            context_window=args.context_window,
        )
        pd.DataFrame(summary).to_csv(args.output / "paper_summary.csv", index=False, encoding="utf-8-sig")
        if len(papers) < 2:
            raise RuntimeError("至少需要两篇含足够可提取文字的 PDF。")

        documents = [p.text for p in papers]
        if custom_keywords is not None and args.keyword_mode == "exact":
            terms, binary, df, mean_tfidf, candidates, excluded_candidates = (
                extract_custom_keywords(documents, custom_keywords)
            )
            merged_terms = terms.tolist()
            merged_binary = binary
            audit = pd.DataFrame({
                "original_keyword": terms,
                "merged_keyword": terms,
                "similarity_to_representative": np.ones(len(terms)),
                "automatically_merged": np.zeros(len(terms), dtype=bool),
                "original_document_frequency": df,
            })
            suggestions = pd.DataFrame(columns=[
                "keyword_1", "keyword_2", "cosine_similarity", "protected_from_merge"
            ])
            merged_similarity = (
                semantic_similarity(merged_terms) if args.group_threshold > 0
                else np.eye(len(merged_terms))
            )
        else:
            terms, binary, df, mean_tfidf, candidates, excluded_candidates = extract_candidates(
                documents, args.top_n, args.min_df, args.max_df, topic
            )
            clusters, similarity, suggestions = semantic_clusters(
                terms, df, args.merge_threshold, args.suggest_threshold, topic
            )
            merged_terms, merged_binary, audit = merge_keyword_matrix(
                terms, binary, df, mean_tfidf, clusters, similarity
            )
            original_index = {str(term): index for index, term in enumerate(terms)}
            representative_indices = [original_index[term] for term in merged_terms]
            merged_similarity = similarity[np.ix_(representative_indices, representative_indices)]
        group_assignments, group_table = semantic_groups(
            merged_terms, merged_similarity, args.group_threshold
        )
        graph, cooccurrence = build_network(
            merged_terms, merged_binary, args.min_edge,
            args.min_association, args.max_edges_per_node, group_assignments,
        )
        save_results(args.output, papers, summary, candidates, excluded_candidates,
                     merged_terms, merged_binary,
                     cooccurrence, audit, suggestions, group_table, graph)
        draw_network(graph, args.output, args.seed, args.show)
        write_interactive_network(graph, args.output, args.seed)

        merged_count = len(terms) - len(merged_terms)
        if custom_keywords is not None and args.keyword_mode == "exact":
            found_count = int((merged_binary.sum(axis=0) > 0).sum())
            print(f"\n完成：纳入 {len(papers)} 篇论文，自定义关键词 "
                  f"{len(merged_terms)} 个，其中 {found_count} 个在论文中出现。")
        elif guided:
            print(f"\n完成：纳入 {len(papers)} 篇主题相关论文，"
                  f"由 {len(custom_keywords)} 个用户词引导发现 {len(merged_terms)} 个概念。")
        else:
            print(f"\n完成：纳入 {len(papers)} 篇论文，候选词 {len(terms)} 个，"
                  f"自动合并 {merged_count} 个，最终概念 {len(merged_terms)} 个。")
        print(f"网络：{graph.number_of_nodes()} 个节点，{graph.number_of_edges()} 条边。")
        print(f"语义分组：{group_table['semantic_group'].nunique()} 组，"
              f"共 {len(group_table)} 个被标记节点。")
        print(f"结果目录：{args.output.resolve()}")
        return 0
    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
