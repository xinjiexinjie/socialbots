# -*- coding: utf-8 -*-
"""
重构版 Step 2：多模态聚类与自动标记“高疑似”评论
依赖：
  pip install pandas scikit-learn matplotlib seaborn
说明：
  1. 动态选择最优聚类数（基于 Silhouette 分数）
  2. 聚类后自动根据“可疑度评分”对簇进行排序，
     将均值最高的 25% 簇标记为“高疑似”，
     将均值最低的 25% 簇标记为“正常评论”，
     中间簇保持“中度疑似”。
  3. 保存带更新“初步分类”标签的结果。
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. 读取 Step1 输出结果
# ----------------------------
INPUT_PATH = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/comment_scoring_result.csv'
df = pd.read_csv(INPUT_PATH, encoding='utf-8')

# ----------------------------
# 2. 筛选“中度疑似”子集
# ----------------------------
df_mid = df[df['初步分类'] == '中度疑似'].copy()
if df_mid.empty:
    raise ValueError("没有找到“中度疑似”样本，请确认 Step1 是否正确标记。")

# ----------------------------
# 3. 构造用于聚类的特征矩阵
# ----------------------------
features = [
    'length',            # 评论长度
    '点赞数',            # 点赞数
    'ttr',               # 词汇多样性
    'punct_ratio',       # 标点丰富度
    'dup_score',         # 文本重复频次
    'multi_reply_score', # 同用户同帖多评次数
    'ad_flag'            # 广告词命中标志 (0/1)
]
X = df_mid[features].astype(float).fillna(0)

# ----------------------------
# 4. 特征标准化
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 5. 自动选择最佳 k（基于 Silhouette 分数）
# ----------------------------
sil_scores = []
candidate_ks = range(2, 20)  # 可根据需要调整范围
for k in candidate_ks:
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

# 找到使 Silhouette 分数最大的 k
best_k = candidate_ks[int(np.argmax(sil_scores))]
print(f"自动选出的最佳聚类数 k = {best_k}（Silhouette 分数 = {max(sil_scores):.4f}）")

# 可视化 Silhouette 分数曲线（可选）
plt.figure(figsize=(6, 4))
plt.plot(list(candidate_ks), sil_scores, '-o')
plt.xlabel('聚类数 k')
plt.ylabel('Silhouette 分数')
plt.title('Silhouette 分析：确定最佳 k')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 6. 用 KMeans 聚类
# ----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42)
df_mid['Step2_cluster'] = kmeans.fit_predict(X_scaled)

# 把聚类标签同步回原始 df
df.loc[df_mid.index, 'Step2_cluster'] = df_mid['Step2_cluster']

# ----------------------------
# 7. 按簇计算“可疑度评分”均值，并排序
# ----------------------------
# 确保 df_mid 中含有 '可疑度评分' 列（来自 Step1）
cluster_score_means = (
    df_mid.groupby('Step2_cluster')['可疑度评分']
    .mean()
    .rename('cluster_mean_score')
)
# 合并回 df_mid
df_mid = df_mid.merge(cluster_score_means, left_on='Step2_cluster', right_index=True)

# 统计所有簇的均值分布
cluster_means = cluster_score_means.sort_values(ascending=False)
num_clusters = len(cluster_means)

# 定义阈值：取上 25% 均值簇为“高疑似”，下 25% 为“正常评论”，其余保留“中度疑似”
high_cutoff = int(np.ceil(num_clusters * 0.25))
low_cutoff = int(np.floor(num_clusters * 0.25))

high_clusters = cluster_means.index[:high_cutoff].tolist()
low_clusters  = cluster_means.index[-low_cutoff:].tolist()

print(f"均值最高的 {high_cutoff} 个簇标为“高疑似”：{high_clusters}")
print(f"均值最低的 {low_cutoff} 个簇标为“正常评论”：{low_clusters}")

# ----------------------------
# 8. 根据簇标签更新“初步分类”
# ----------------------------
def update_label(row):
    c = row['Step2_cluster']
    if c in high_clusters:
        return '高疑似'
    elif c in low_clusters:
        return '正常评论'
    else:
        return '中度疑似'

df_mid['初步分类_updated'] = df_mid.apply(update_label, axis=1)
df.loc[df_mid.index, '初步分类'] = df_mid['初步分类_updated']

# ----------------------------
# 9. 保存更新后的结果
# ----------------------------
OUTPUT_PATH = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/comment_scoring_step3_result.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print("Step2 完成，已保存至：", OUTPUT_PATH)
print("更新后各分类计数：")
print(df['初步分类'].value_counts())

# ----------------------------
# 10. （可选）PCA 可视化聚类分布
# ----------------------------
pca = PCA(n_components=2, random_state=42)
pc = pca.fit_transform(X_scaled)
df_mid['pca1'] = pc[:, 0]
df_mid['pca2'] = pc[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='pca1', y='pca2',
    hue='Step2_cluster',
    palette='Set2',
    data=df_mid,
    legend='full',
    alpha=0.7
)
plt.title('')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

SAVE_PNG = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/step2_pca_clusters.png'
plt.savefig(SAVE_PNG, dpi=300)
plt.show()
print(f"PCA 可视化已保存至：{SAVE_PNG}")