# -*- coding: utf-8 -*-
"""
Revised Step 3: 高疑似用户共现 & 文本相似网络分析（自动化、避免全量 O(N^2) 计算）
依赖：
  pip install pandas networkx python-louvain scikit-learn sentence-transformers matplotlib seaborn
说明：
  1. 从 Step2 标记为“高疑似”的评论中提取用户列表。
  2. 构建“笔记ID → 用户集合”映射，计算用户–用户的评论共现次数。
  3. 基于共现次数的分布自动选取 MIN_COOC（例如 90% 分位），只保留最强的共现边。
  4. 计算每个用户的文本级别嵌入，并用近邻检索 (NearestNeighbors) 而非全量矩阵来筛选高相似度对。
  5. 构建三个子图：仅共现 (G_cooc)、仅文本相似 (G_text)、及交集 (G_both)，并执行 Louvain 社区发现。
  6. 导出社区统计、用户层面行为/文本度数，并按社区分组导出高疑似评论。
"""

import os
import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from itertools import combinations

from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
import community as community_louvain

# ----------------------------
# 1. 输入 / 输出 配置
# ----------------------------
INPUT_CSV = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/comment_scoring_step3_result.csv'
OUTPUT_DIR = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/step3_network'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 2. 读取 & 筛选“高疑似”评论
# ----------------------------
df = pd.read_csv(INPUT_CSV, encoding='utf-8')
df_high = df[df['初步分类'] == '高疑似'].dropna(subset=['用户ID', '评论内容']).reset_index(drop=True)
print(f"Step3 开始：高疑似评论共 {len(df_high)} 条")

# ----------------------------
# 3. 构建“笔记ID → 用户集合”映射 & 计算共现次数
# ----------------------------
note_to_users = defaultdict(set)
for _, row in df_high.iterrows():
    note_to_users[row['笔记ID']].add(str(row['用户ID']))

co_occurrence = defaultdict(int)
for users in note_to_users.values():
    for u1, u2 in combinations(sorted(users), 2):
        co_occurrence[(u1, u2)] += 1

# 若没有共现对则退出
if not co_occurrence:
    raise ValueError("未发现任何用户–用户共现关系，请检查“高疑似”样本或数据分布。")

# ----------------------------
# 4. 动态选择 MIN_COOC：仅保留共现次数排名前 10% 的边
# ----------------------------
co_values = np.array(list(co_occurrence.values()))
MIN_COOC = int(np.percentile(co_values, 90))
if MIN_COOC < 1:
    MIN_COOC = 1
print(f"动态选取 MIN_COOC = {MIN_COOC}（共现次数 ≥ {MIN_COOC}）")

# ----------------------------
# 5. 计算用户级别文本嵌入 & 使用 NearestNeighbors 筛选高相似度边
# ----------------------------
# 5.1 提取所有“高疑似”评论对应的文本列表
texts = df_high['评论内容'].tolist()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
comment_embs = model.encode(texts, convert_to_tensor=True).cpu().numpy()

# 5.2 计算每个用户的平均嵌入（user_embs）
comment_users = df_high['用户ID'].astype(str).tolist()
user_to_idxs = defaultdict(list)
for idx, uid in enumerate(comment_users):
    user_to_idxs[uid].append(idx)
user_ids = list(user_to_idxs.keys())

# stack 平均
user_embs = np.vstack([
    comment_embs[user_to_idxs[uid]].mean(axis=0)
    for uid in user_ids
])
# 归一化（方便余弦距离计算）
user_norms = np.linalg.norm(user_embs, axis=1, keepdims=True)
user_embs_normed = user_embs / np.clip(user_norms, a_min=1e-9, a_max=None)

# 5.3 随机采样若干对以估计相似度阈值（TEXT_SIM_THRESHOLD）
num_users = len(user_ids)
if num_users <= 1000:
    # 样本量较小时直接计算全量上三角相似度
    sim_mat = np.matmul(user_embs_normed, user_embs_normed.T)
    sims = sim_mat[np.triu_indices(num_users, k=1)]
else:
    # 样本量大时随机采样 10000 对
    sims = []
    all_indices = list(range(num_users))
    for _ in range(10000):
        i, j = random.sample(all_indices, 2)
        sims.append(np.dot(user_embs_normed[i], user_embs_normed[j]))
    sims = np.array(sims)

# 选择 90% 分位作为相似度阈值
TEXT_SIM_THRESHOLD = float(np.percentile(sims, 90))
print(f"动态选取 TEXT_SIM_THRESHOLD = {TEXT_SIM_THRESHOLD:.4f}（保留最强 10% 相似度边）")

# 5.4 使用 NearestNeighbors 构建高相似度边 (cosine)：
# “distance = 1 - cosine_similarity”，因此要保留 distance ≤ (1 - threshold)
nbrs = NearestNeighbors(metric='cosine', algorithm='auto')
nbrs.fit(user_embs_normed)
# children: 对每个节点找 50 个最近邻（足够候选），再筛 threshold
distances, indices = nbrs.kneighbors(user_embs_normed, n_neighbors=50, return_distance=True)

G_text = nx.Graph()
G_text.add_nodes_from(user_ids)

for i, uid in enumerate(user_ids):
    for dist, j in zip(distances[i], indices[i]):
        if i == j:
            continue
        sim = 1 - dist
        if sim >= TEXT_SIM_THRESHOLD:
            u1, u2 = uid, user_ids[j]
            # 添加无向边
            G_text.add_edge(u1, u2, text_sim=sim)

# ----------------------------
# 6. 构建三种子图：G_cooc、G_text、G_both
# ----------------------------
# 6.1 G_cooc：仅行为共现（共现次数 ≥ MIN_COOC）
G_cooc = nx.Graph()
G_cooc.add_nodes_from(user_ids)
for (u1, u2), w in co_occurrence.items():
    if w >= MIN_COOC:
        G_cooc.add_edge(u1, u2, weight=w)

# 6.2 G_both：行为共现 & 文本相似交集
G_both = nx.Graph()
G_both.add_nodes_from(user_ids)
for (u1, u2), w in co_occurrence.items():
    if w >= MIN_COOC and G_text.has_edge(u1, u2):
        sim = G_text[u1][u2]['text_sim']
        G_both.add_edge(u1, u2, weight=w, text_sim=sim)

# ----------------------------
# 7. Louvain 社区发现（基于 G_both 加权）
# ----------------------------
if G_both.number_of_edges() == 0:
    raise ValueError("G_both 没有任何边，无法进行社区检测，请检查阈值设置或数据。")

ALPHA = 5.0
for u, v, a in G_both.edges(data=True):
    a['combined_weight'] = a['weight'] + ALPHA * a['text_sim']

partition_both = community_louvain.best_partition(G_both, weight='combined_weight', resolution=0.5)
nx.set_node_attributes(G_cooc, partition_both, 'community_both')
nx.set_node_attributes(G_text, partition_both, 'community_both')
nx.set_node_attributes(G_both, partition_both, 'community_both')


# ----------------------------
# 8. 导出社区统计 & 用户度数统计
# ----------------------------
# 8.1 社区 → 节点列表映射
comm_nodes = defaultdict(list)
for node, comm in partition_both.items():
    comm_nodes[comm].append(node)

comm_stats = []
for comm, nodes in comm_nodes.items():
    text_sim_degrees = [
        sum(a.get('text_sim', 0.0) for _, _, a in G_both.edges(n, data=True))
        for n in nodes
    ]
    avg_sim = float(np.mean(text_sim_degrees)) if text_sim_degrees else 0.0
    max_sim = float(np.max(text_sim_degrees)) if text_sim_degrees else 0.0
    med_sim = float(np.median(text_sim_degrees)) if text_sim_degrees else 0.0

    comm_stats.append({
        'community': comm,
        'num_users': len(nodes),
        'avg_text_sim_degree': avg_sim,
        'max_text_sim_degree': max_sim,
        'median_text_sim_degree': med_sim
    })

pd.DataFrame(comm_stats).to_csv(
    os.path.join(OUTPUT_DIR, 'community_textsim_strength_both.csv'),
    index=False, encoding='utf-8-sig'
)

# 8.2 用户层面 text_sim_degree 与 behavior_degree
user_textsim_degree = {}
user_behavior_degree = {}
for u in G_both.nodes():
    ts_deg = sum(a.get('text_sim', 0.0) for _, _, a in G_both.edges(u, data=True))
    bh_deg = sum(a.get('weight', 0) for _, _, a in G_both.edges(u, data=True))
    user_textsim_degree[u] = float(ts_deg)
    user_behavior_degree[u] = int(bh_deg)

user_stats_df = pd.DataFrame({
    '用户ID': list(user_textsim_degree.keys()),
    'text_sim_degree': list(user_textsim_degree.values()),
    'behavior_degree': list(user_behavior_degree.values())
})
user_stats_df['community_both'] = user_stats_df['用户ID'].map(partition_both)
user_stats_df.to_csv(
    os.path.join(OUTPUT_DIR, 'user_textsim_strength_both.csv'),
    index=False, encoding='utf-8-sig'
)

from scipy.stats import pearsonr

# 从两个图中获取所有节点
common_users = list(set(G_cooc.nodes()) & set(G_text.nodes()))

# 分别获取度数
behavior_degrees = [G_cooc.degree(u, weight='weight') for u in common_users]
textsim_degrees = [G_text.degree(u, weight='text_sim') for u in common_users]

# 计算皮尔逊相关
corr, pval = pearsonr(behavior_degrees, textsim_degrees)
print(f"Pearson r: r = {corr:.4f}, p = {pval:.4g}")


# ----------------------------
# 9. 导出高疑似评论按社区分组（df_high enriched with text/behavior度数）
# ----------------------------
df_high['community_both'] = df_high['用户ID'].astype(str).map(partition_both)
df_high = df_high.merge(
    user_stats_df[['用户ID', 'text_sim_degree', 'behavior_degree']],
    on='用户ID', how='left'
).sort_values(by=['community_both', '用户ID']).reset_index(drop=True)

export_path = os.path.join(OUTPUT_DIR, 'high_suspect_comments_by_community.csv')
df_high.to_csv(export_path, index=False, encoding='utf-8-sig')
print(f"高疑似评论按社区分组已导出：{export_path}")

# ----------------------------
# 10. 可视化三种子图
# ----------------------------
def plot_graph(H, title, filename):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(H, seed=42, k=0.15)
    node_colors = [H.nodes[n].get('community_both', 0) for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos,
                           node_size=50,
                           node_color=node_colors,
                           cmap=cm.Set2,
                           alpha=0.8)
    widths = [H[u][v].get('weight', 1) * 0.1 if 'weight' in H[u][v] else 0.1
              for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, width=widths, edge_color='lightgray', alpha=0.4)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.show()

plot_graph(G_cooc, f"", f"network_cooc_ge_{MIN_COOC}.png")
plot_graph(G_text, f"", f"network_textsim_ge_{int(TEXT_SIM_THRESHOLD*100)}.png")
plot_graph(G_both, f"", f"network_both_cooc_{MIN_COOC}_textsim.png")

print("Step 3 完成，所有结果已保存至", OUTPUT_DIR)