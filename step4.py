# -*- coding: utf-8 -*-
"""
Revised Step 4: LDA 主题建模与情绪分析（自动选择主题数）
依赖：
  pip install pandas jieba numpy scikit-learn snownlp matplotlib seaborn
说明：
  1. 清洗所有评论，分词并构建文档-词矩阵（CountVectorizer + 二值化共现矩阵）。
  2. 计算一系列候选主题数的 Perplexity 和 UMass 一致性，并自动选出最佳主题数。
  3. 用选出的主题数训练最终 LDA 模型，打印主题关键词和典型评论。
  4. 对所有评论进行情绪分析（SnowNLP），并可视化各主题的情绪分布和评论数量。
  5. 保存含“Dominant_Topic”和“Sentiment_Score”的结果到 CSV。
"""

import os
import re
import string
import pandas as pd
import numpy as np
import jieba

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP

# ────────────────────────────────────────────────────────────────────────
# 1. 读取原始评论并清洗
# ────────────────────────────────────────────────────────────────────────
INPUT_CSV = '/Users/xinjie/PycharmProjects/socialbots/data_cleaned.csv'
df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
df = df.dropna(subset=['评论内容']).reset_index(drop=True)

def clean_noise(text: str) -> str:
    """
    1. 提取 “[xxxR]” 表情占位，用 __EMOJIi__ 代替
    2. 删除 ASCII 标点、常见中文标点、英文字母、数字、@
    3. 恢复 “[xxxR]” 表情
    """
    emoji_pattern = re.compile(r'\[[^\]]*R\]')
    emojis = emoji_pattern.findall(text)
    for idx, em in enumerate(emojis):
        placeholder = f" __EMOJI{idx}__ "
        text = text.replace(em, placeholder)

    # 删除 ASCII 标点
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    # 删除常见中文标点
    chinese_punct = "，。！？；：“”‘’、《》……—·"
    text = re.sub(rf"[{chinese_punct}]", "", text)
    # 删除英文字母、数字、@ 符号
    text = re.sub(r'[A-Za-z0-9@]', '', text)

    for idx, em in enumerate(emojis):
        placeholder = f"__EMOJI{idx}__"
        text = text.replace(placeholder, em)

    return text

def jieba_tokenizer(text: str) -> list:
    """
    分词逻辑：
    1. 将 “[xxxR]” 表情替换成 __EMOJIi__ 占位
    2. jieba.lcut 分词
    3. 如果 token 是占位符，就还原为 “[xxxR]”；否则只保留长度 ≥ 2 且含汉字的 token
    """
    emoji_pattern = re.compile(r'\[[^\]]*R\]')
    emojis = emoji_pattern.findall(text)
    for idx, em in enumerate(emojis):
        text = text.replace(em, f" __EMOJI{idx}__ ")

    raw_tokens = jieba.lcut(text)
    clean_tokens = []
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue
        if re.fullmatch(r"__EMOJI\d+__", tok):
            idx = int(tok.replace("__EMOJI", "").replace("__", ""))
            clean_tokens.append(emojis[idx])
            continue
        if len(tok) < 2:
            continue
        if not re.search(r'[\u4e00-\u9fff]', tok):
            continue
        clean_tokens.append(tok)
    return clean_tokens

# 定义停用词（可根据需要调整）
my_stopwords = [
    '的','了','我','是','你','不','都','也','就','给','好','有','啊','我们',
    '哈哈哈','哈哈哈哈','the','doge','哈哈',
    '就是','这个','什么','现在','可以','他们','一个','真的',
    '谢谢','加油','你们','不是','怎么','可能','还是',
    '好好','太好了','啊啊啊','老师','教授','没有','自己','知道',
    '这样','这么','但是','已经','评论','一起','今天','因为','只是',
    '还有','这些','应该','意思','很多','不会','喜欢','爱看','世界',
    '国家','以后','一下','时候','感觉','大家','意义','原来','这种',
    '发现','不行','不要','为什么','所以','是不是','直接','一样',
    '红薯','好像','一次','注意','看到','最新','开始','发下','看见',
    '留下','好看','朋友','小红书','觉得','其实','不能','那个','咱们',
    '东西','出现','存在','讨论','点赞','看看','需要','突然','还在',
    '一直','真正','明白','话题','有点','的话','出来','人家','真的','想不想',
    '以为','如果','你好'
]

# 清洗并生成“Cleaned”列
df['Cleaned'] = df['评论内容'].astype(str).map(clean_noise)

# 分词并拼接成“文档”列表
tokenized_texts = [jieba_tokenizer(t) for t in df['Cleaned']]
docs_joined = [" ".join(tokens) for tokens in tokenized_texts]

# ────────────────────────────────────────────────────────────────────────
# 2. 构造文档-词矩阵：CountVectorizer + 二值化矩阵
# ────────────────────────────────────────────────────────────────────────
count_vec = CountVectorizer(
    tokenizer=lambda x: x.split(),
    max_df=0.8,
    min_df=5,
    stop_words=my_stopwords
)
X_counts = count_vec.fit_transform(docs_joined)  # n_docs × n_vocab
vocab = count_vec.get_feature_names_out().tolist()
word2index = {w: i for i, w in enumerate(vocab)}

bin_vec = CountVectorizer(
    vocabulary=count_vec.vocabulary_,  # 保持同一词表
    tokenizer=lambda x: x.split(),
    binary=True
)
X_bin = bin_vec.fit_transform(docs_joined)  # 二值化矩阵
D_w = np.array(X_bin.sum(axis=0)).ravel()    # 每个词出现的文档数

# ────────────────────────────────────────────────────────────────────────
# 3. 定义 UMass 一致性计算函数
# ────────────────────────────────────────────────────────────────────────
def compute_umass_for_topic(top_words, X_bin_matrix, D_w_array, w2i):
    """
    计算一个主题的 UMass coherence:
    C = sum_{i=2..K} sum_{j=1..i-1} log( [D(w_i, w_j) + 1] / D(w_j) )
    """
    umass_score = 0.0
    K = len(top_words)
    idx_list = [w2i[w] for w in top_words]
    X_bin_bool = (X_bin_matrix.toarray() > 0)  # 转为布尔矩阵
    for i in range(1, K):
        for j in range(i):
            idx_i = idx_list[i]
            idx_j = idx_list[j]
            D_ij = int(np.logical_and(X_bin_bool[:, idx_i], X_bin_bool[:, idx_j]).sum())
            D_j = D_w_array[idx_j]
            umass_score += np.log((D_ij + 1) / D_j)
    return umass_score

# ────────────────────────────────────────────────────────────────────────
# 4. 训练不同主题数，记录 Perplexity 与 UMass Coherence
# ────────────────────────────────────────────────────────────────────────
topic_range = list(range(1, 21))
perplexities = []
coherences_umass = []

for n_topics in topic_range:
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='batch',
        random_state=42
    )
    lda_model.fit(X_counts)

    perp = lda_model.perplexity(X_counts)
    perplexities.append(perp)

    top_words_for_all_topics = []
    for comp in lda_model.components_:
        top_indices = comp.argsort()[::-1][:10]
        top_words = [vocab[i] for i in top_indices]
        top_words_for_all_topics.append(top_words)

    umass_list = [
        compute_umass_for_topic(top_words, X_bin, D_w, word2index)
        for top_words in top_words_for_all_topics
    ]
    coherences_umass.append(np.mean(umass_list))

# ────────────────────────────────────────────────────────────────────────
# 5. 可视化 Perplexity vs UMass Coherence，并自动选出最佳主题数
# ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(topic_range, perplexities, '-o')
plt.title('困惑度 vs 主题数')
plt.xlabel('主题数')
plt.ylabel('Perplexity')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(topic_range, coherences_umass, '-o', color='orange')
plt.title('UMass 一致性 vs 主题数')
plt.xlabel('主题数')
plt.ylabel('UMass Coherence')
plt.grid(True)

plt.tight_layout()
plt.show()

# 自动选出 UMass Coherence 最大对应的主题数
best_idx = int(np.argmax(coherences_umass))
n_final = topic_range[best_idx]
print(f"\n自动选择最佳主题数 n_final = {n_final} (最大 UMass: {coherences_umass[best_idx]:.4f})")

# ────────────────────────────────────────────────────────────────────────
# 6. 用选出的主题数训练最终 LDA 模型
# ────────────────────────────────────────────────────────────────────────
lda_final = LatentDirichletAllocation(
    n_components=n_final,
    max_iter=10,
    learning_method='batch',
    random_state=42
)
lda_matrix = lda_final.fit_transform(X_counts)  # n_docs × n_final

# 打印每个主题的 Top 10 关键词
print(f"\n=== 最终选用 {n_final} 个主题，Top 10 关键词如下： ===")
for topic_idx, comp in enumerate(lda_final.components_):
    top_indices = comp.argsort()[::-1][:10]
    top_words = [vocab[i] for i in top_indices]
    print(f"主题 {topic_idx}: {top_words}")

# 为每条评论标记“Dominant_Topic”
df['Dominant_Topic'] = lda_matrix.argmax(axis=1)

# 打印每个主题下概率最高的前 3 条“典型评论”
print("\n=== 每个主题下 Top 3 典型评论 ===")
for topic_idx in range(n_final):
    probs = lda_matrix[:, topic_idx]
    top3_idxs = probs.argsort()[::-1][:3]
    print(f"\n--- 主题 {topic_idx} ---")
    for rank, doc_idx in enumerate(top3_idxs, start=1):
        print(f"排名 {rank}: {df.loc[doc_idx, '评论内容']}")

# ────────────────────────────────────────────────────────────────────────
# 7. 情绪分析（SnowNLP） & 可视化
# ────────────────────────────────────────────────────────────────────────
sentiment_scores = []
for text in docs_joined:
    txt = text.strip()
    if not txt:
        sentiment_scores.append(0.5)
    else:
        try:
            sentiment_scores.append(SnowNLP(txt).sentiments)
        except:
            sentiment_scores.append(0.5)

df['Sentiment_Score'] = sentiment_scores

plt.figure(figsize=(8, 5))
sns.histplot(sentiment_scores, bins=20, kde=True)
plt.title('')
plt.xlabel('Sentiment Score (0–1)')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.savefig('/Users/xinjie/PycharmProjects/socialbots/analysis_results/sentiment_score_distribution.png', dpi=300)
plt.show()

topic_sentiment = df.groupby('Dominant_Topic')['Sentiment_Score'].mean().reset_index()
print("\n=== 各主题的平均情绪得分 ===")
print(topic_sentiment)

topic_counts = df['Dominant_Topic'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='Set2')
plt.title('')
plt.xlabel('Topic Number')
plt.ylabel('Number of Comments')
plt.savefig('/Users/xinjie/PycharmProjects/socialbots/analysis_results/topic_comment_counts.png', dpi=300)
plt.tight_layout()

plt.show()

# ────────────────────────────────────────────────────────────────────────
# 8. 保存分析结果到 CSV
# ────────────────────────────────────────────────────────────────────────
OUTPUT_PATH = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/step4_comment_topic_sentiment.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df[['评论ID', '评论内容', 'Dominant_Topic', 'Sentiment_Score']].to_csv(
    OUTPUT_PATH, index=False, encoding='utf-8-sig'
)
print(f"\n分析结果已保存至：{OUTPUT_PATH}")