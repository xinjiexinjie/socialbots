# -*- coding: utf-8 -*-
"""
重构版 Step 1：动态阈值 + 内容质量特征异常打分脚本
功能：
 1. 自动根据数据分布（拐点/分位）选阈值
 2. 剔除“短评论”特征，新增词汇多样性和标点丰富度特征
 3. 做阈值敏感性分析
 4. 使用 Snorkel 弱监督学习各特征权重
 5. 基于学习到的权重给每条评论打分并动态设定“中度/高疑似”阈值
依赖：
 pip install pandas numpy kneed jieba snorkel
"""

import os
import re
import string
import pandas as pd
import numpy as np
import jieba
from kneed import KneeLocator
import matplotlib.pyplot as plt
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

# ----------------------------
# 1. 文件路径配置
# ----------------------------
INPUT_CSV    = '/Users/xinjie/PycharmProjects/socialbots/data_cleaned.csv'
KEYWORD_FILE = '/Users/xinjie/PycharmProjects/socialbots/ad_keywords.txt'
OUTPUT_CSV   = '/Users/xinjie/PycharmProjects/socialbots/analysis_results/comment_scoring_result.csv'

# ----------------------------
# 2. 加载原始数据
# ----------------------------
df = pd.read_csv(
    INPUT_CSV,
    encoding='utf-8',
    dtype={'评论ID': str, '用户名称': str, '用户ID': str, '笔记ID': str},
    low_memory=False
)
df['评论内容'] = df['评论内容'].fillna('').astype(str)
df['用户名称'] = df['用户名称'].fillna('').astype(str)

# ----------------------------
# 3. 加载广告关键词，构建模糊正则
# ----------------------------
try:
    with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
        AD_PATTERNS = [line.strip().lower() for line in f if line.strip()]
    if not AD_PATTERNS:
        raise ValueError
except Exception:
    print("[警告] 加载广告关键词失败，使用内置列表")
    AD_PATTERNS = [
        'follow me', '关注我',
        '微信', 'VX', 'V信', '微X', 'V',
        'QQ', '企鹅号',
        '手机号', '电话', '手机',
        '加我', '私信', '私聊', '联系我', '找我',
        '二维码', '扫码', '扫一扫',
        '链接', '网址', '网页', '点击这里',
        '公众号', '小程序', '店铺名',
        '推荐', '安利', '种草', '回购', '必买', '必备',
        '购买', '下单', '抢购', '秒杀', '限时', '优惠', '折扣',
        '价格', '多少钱', '划算', '便宜', '实惠',
        '代购', '正品', '原单', '高仿', 'A货',
        '淘宝', '天猫', '京东', '拼多多', '苏宁',
        '小程序', 'APP', '下载', '安装',
        '站外', '跳转', '导流', '引流', '外链',
        '私域', '社群', '微信群', 'QQ群',
        '点赞', '收藏', '关注', '转发', '评论', '留言',
        '抽奖', '福利', '赠品', '送礼', '红包',
        '参与', '报名', '预约', '体验',
        '免费', '试用', '体验装', '试吃',
        '赚钱', '兼职', '副业', '暴利', '日入', '月入',
        '减肥', '瘦身', '美白', '祛斑', '祛痘', '丰胸',
        '整形', '医美', '隆鼻', '抽脂',
        '赌博', '博彩', '彩票', '下注'
    ]

def build_fuzzy_pattern(pat: str) -> str:
    return ''.join(re.escape(ch) + r'\s*' for ch in pat).rstrip(r'\s*')

AD_PATTERN_REGEX = re.compile(
    r'(' + '|'.join(build_fuzzy_pattern(p) for p in AD_PATTERNS) + r')',
    flags=re.IGNORECASE
)

# ----------------------------
# 4. 账号用户名正则
# ----------------------------
USERNAME_PATTERN = re.compile(r'^小红薯[0-9A-Za-z]+$')

# ----------------------------
# 5. 处理“点赞数”列（支持“万”单位）
# ----------------------------
def parse_likes(x):
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if s.endswith('万'):
        try:
            return int(float(s[:-1]) * 10000)
        except:
            return 0
    num = re.sub(r'[^\d.]', '', s)
    try:
        return int(float(num))
    except:
        return 0

df['点赞数'] = df['点赞数'].apply(parse_likes)

# ============================
# 以下定义函数，按模块化重构
# ============================

def compute_dynamic_thresholds(df: pd.DataFrame) -> dict:
    """
    计算各内容质量特征的动态阈值：
      - 评论长度：5% 分位
      - 完全重复评论次数：拐点 & 99% 分位融合
      - 同一用户同帖多评次数：拐点 & 99% 分位融合
      - 词汇多样性（TTR）：5% 分位
      - 标点丰富度：5% 分位
    返回字典 keys 包括 LENGTH_THRESHOLD, DUPLICATE_THRESHOLD, MULTI_DUP_THRESHOLD, TTR_THRESHOLD, PUNCT_THRESHOLD, dup_counts, unp_counts
    """
    # 6.1 评论长度 → 5% 分位
    lengths = df['评论内容'].map(len)
    LENGTH_THRESHOLD = int(lengths.quantile(0.05))

    # 6.2 完全重复评论次数：先拐点，再取 99% 分位和拐点的较小者
    dup_counts = df['评论内容'].value_counts()
    dups_sorted = np.sort(dup_counts.values)
    k_dup = KneeLocator(np.arange(len(dups_sorted)), dups_sorted, curve='convex', direction='increasing').knee
    dup_knee = int(dups_sorted[k_dup]) if k_dup is not None else int(dups_sorted.mean())
    dup_99 = int(dup_counts.quantile(0.99))
    DUPLICATE_THRESHOLD = min(dup_knee, dup_99)

    # 6.3 同一用户同帖多评次数：拐点 + 99% 分位
    user_pairs = df[['用户ID','笔记ID']].astype(str).agg('::'.join, axis=1)
    unp_counts = user_pairs.value_counts()
    unp_sorted = np.sort(unp_counts.values)
    k_unp = KneeLocator(np.arange(len(unp_sorted)), unp_sorted, curve='convex', direction='increasing').knee
    unp_knee = int(unp_sorted[k_unp]) if k_unp is not None else int(unp_sorted.mean())
    unp_99 = int(unp_counts.quantile(0.99))
    MULTI_DUP_THRESHOLD = min(unp_knee, unp_99)

    # 6.4 词汇多样性 TTR → 5% 分位
    def tokenize(text: str) -> list:
        return [tok for tok in jieba.lcut(text) if tok.strip()]

    df['_tokens_tmp'] = df['评论内容'].map(tokenize)
    df['ttr'] = df['_tokens_tmp'].map(lambda toks: len(set(toks)) / len(toks) if toks else 0.0)
    TTR_THRESHOLD = df['ttr'].quantile(0.05)

    # 6.5 标点丰富度 → 5% 分位
    PUNCT_SET = set(string.punctuation) | set('！？｡。…、“”‘’')
    df['punct_ratio'] = df['评论内容'].map(
        lambda txt: sum(ch in PUNCT_SET for ch in txt) / len(txt) if txt else 0.0
    )
    PUNCT_THRESHOLD = df['punct_ratio'].quantile(0.05)

    thresholds = {
        'LENGTH_THRESHOLD': LENGTH_THRESHOLD,
        'DUPLICATE_THRESHOLD': DUPLICATE_THRESHOLD,
        'MULTI_DUP_THRESHOLD': MULTI_DUP_THRESHOLD,
        'TTR_THRESHOLD': TTR_THRESHOLD,
        'PUNCT_THRESHOLD': PUNCT_THRESHOLD,
        'dup_counts': dup_counts,
        'unp_counts': unp_counts
    }
    df.drop(columns=['_tokens_tmp'], inplace=True)
    return thresholds

def print_threshold_sensitivity(df: pd.DataFrame, thresholds: dict):
    """
    对 50 个 bootstrap 样本重复计算阈值，用 describe 打印 min/25%/50%/75%/max 范围，并保存到 CSV。
    """
    def sample_once(sub_df):
        lengths = sub_df['评论内容'].map(len)
        lt = int(lengths.quantile(0.05))

        dc = sub_df['评论内容'].value_counts()
        dups_sorted = np.sort(dc.values)
        k_dup = KneeLocator(np.arange(len(dups_sorted)), dups_sorted, curve='convex', direction='increasing').knee
        dup_val = int(dups_sorted[k_dup]) if k_dup is not None and len(dups_sorted)>0 else 1
        dup_99 = int(dc.quantile(0.99))
        dt = min(dup_val, dup_99)

        up = sub_df[['用户ID','笔记ID']].astype(str).agg('::'.join, axis=1).value_counts()
        ups_sorted = np.sort(up.values)
        k_unp = KneeLocator(np.arange(len(ups_sorted)), ups_sorted, curve='convex', direction='increasing').knee
        unp_val = int(ups_sorted[k_unp]) if k_unp is not None and len(ups_sorted)>0 else 1
        unp_99 = int(up.quantile(0.99))
        ut = min(unp_val, unp_99)

        sub_df['tokens_tmp'] = sub_df['评论内容'].map(lambda x: [tok for tok in jieba.lcut(x) if tok.strip()])
        sub_df['ttr_tmp'] = sub_df['tokens_tmp'].map(lambda toks: len(set(toks))/len(toks) if toks else 0.0)
        ttr_ = sub_df['ttr_tmp'].quantile(0.05)

        PUNCT_SET = set(string.punctuation) | set('！？｡。…、“”‘’')
        sub_df['punct_tmp'] = sub_df['评论内容'].map(
            lambda txt: sum(ch in PUNCT_SET for ch in txt)/len(txt) if txt else 0.0
        )
        pr_ = sub_df['punct_tmp'].quantile(0.05)

        sub_df.drop(columns=['tokens_tmp','ttr_tmp','punct_tmp'], inplace=True)
        return {'length': lt, 'duplicate': dt, 'multi_dup': ut, 'ttr': ttr_, 'punct': pr_}

    records = []
    for i in range(50):
        sub = df.sample(frac=0.8, replace=True, random_state=i)
        records.append(sample_once(sub))
    sens_df = pd.DataFrame(records)
    desc = sens_df.describe().loc[['min','25%','50%','75%','max']]
    print("敏感性分析—阈值分位数范围：")
    print(desc)
    sens_df.to_csv(OUTPUT_CSV.replace('.csv','_sens.csv'), index=False)

def train_snorkel_weights(df: pd.DataFrame, thresholds: dict) -> dict:
    """
    使用 Snorkel 弱监督，定义 Labeling Functions 给“可疑=BOT”“正常=NOT_BOT”进行弱标签，
    训练 LabelModel，调用 get_weights() 并根据维度计算“w_BOT - w_NOT_BOT”差值，归一化到 [1,10]。
    返回 learned_weights 字典。
    """
    dup_counts = thresholds['dup_counts']
    unp_counts = thresholds['unp_counts']
    TTR_THRESHOLD = thresholds['TTR_THRESHOLD']
    PUNCT_THRESHOLD = thresholds['PUNCT_THRESHOLD']
    DUPLICATE_THRESHOLD = thresholds['DUPLICATE_THRESHOLD']
    MULTI_DUP_THRESHOLD = thresholds['MULTI_DUP_THRESHOLD']

    ABSTAIN, BOT, NOT_BOT = -1, 1, 0

    # 定义 Labeling Functions
    def lf_ad_phrase(x):       return BOT if AD_PATTERN_REGEX.search(x['评论内容']) else ABSTAIN
    def lf_duplicate(x):       return BOT if dup_counts.get(x['评论内容'], 0) >= DUPLICATE_THRESHOLD else ABSTAIN
    def lf_suspicious_user(x): return BOT if USERNAME_PATTERN.match(x['用户名称']) else ABSTAIN
    def lf_multi_dups(x):      return BOT if unp_counts.get(f"{x['用户ID']}::{x['笔记ID']}", 0) >= MULTI_DUP_THRESHOLD else ABSTAIN
    def lf_low_diversity(x):   return BOT if x['ttr'] < TTR_THRESHOLD else ABSTAIN
    def lf_low_punct(x):       return BOT if x['punct_ratio'] < PUNCT_THRESHOLD else ABSTAIN
    def lf_has_link(x):        return NOT_BOT if "http" in x['评论内容'] else ABSTAIN
    def lf_question(x):
        pattern = r'[？\?]|为什么|怎么|吗|什么'
        return NOT_BOT if re.search(pattern, x['评论内容']) else ABSTAIN
    def lf_emotion_words(x):
        emotion_list = ['难受', '开心', '愤怒', '难过', '伤心', '喜欢', '讨厌']
        return NOT_BOT if any(word in x['评论内容'] for word in emotion_list) else ABSTAIN

    lfs = [
        LabelingFunction("ad_phrase",       lf_ad_phrase),
        LabelingFunction("duplicate",       lf_duplicate),
        LabelingFunction("suspicious_user", lf_suspicious_user),
        LabelingFunction("multi_dups",      lf_multi_dups),
        LabelingFunction("low_diversity",   lf_low_diversity),
        LabelingFunction("low_punct",       lf_low_punct),
        LabelingFunction("has_link",        lf_has_link),
        LabelingFunction("question",        lf_question),
        LabelingFunction("emotion_words",   lf_emotion_words),
    ]

    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df)
    LFAnalysis(L_train, lfs).lf_summary()

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, seed=42)

    # 使用 get_weights()，根据输出维度计算差值
    wm = label_model.get_weights()  # 可能是一维或二维
    if np.ndim(wm) == 2:
        # wm 形状 (2, n_lf)：第一行 for NOT_BOT，第二行 for BOT
        raw_diff = wm[1, :] - wm[0, :]
    elif np.ndim(wm) == 1:
        # wm 形状 (n_lf,)：视为 BOT 权重，NOT_BOT 权重 = 0
        raw_diff = wm.copy()
    else:
        # 单个值
        raw_diff = np.array([float(wm)] * len(lfs))

    raw_clipped = np.clip(raw_diff, a_min=0, a_max=None)
    max_diff = raw_clipped.max() if raw_clipped.max() > 0 else 1.0
    norm_vals = (raw_clipped / max_diff) * 10

    learned_weights = {
        lf.name: max(1, int(round(norm_vals[i])))
        for i, lf in enumerate(lfs)
    }
    print(">> Learned weights:", learned_weights)
    return learned_weights

def score_and_label(df: pd.DataFrame, thresholds: dict, learned_weights: dict) -> pd.DataFrame:
    """
    给每条评论计算“可疑度评分”，并根据分数分布动态设置中度/高疑似阈值：
      - MED_THR = 75% 分位
      - HIGH_THR = 90% 分位
    绘制直方图并打标签，同时补齐 Step 2 所需特征列。
    """
    dup_counts = thresholds['dup_counts']
    unp_counts = thresholds['unp_counts']
    TTR_THRESHOLD = thresholds['TTR_THRESHOLD']
    PUNCT_THRESHOLD = thresholds['PUNCT_THRESHOLD']
    DUPLICATE_THRESHOLD = thresholds['DUPLICATE_THRESHOLD']
    MULTI_DUP_THRESHOLD = thresholds['MULTI_DUP_THRESHOLD']

    def score_row(row):
        text = row['评论内容']
        s, reasons = 0, []
        if AD_PATTERN_REGEX.search(text):
            s += learned_weights['ad_phrase'];       reasons.append('ad_phrase')
        if dup_counts.get(text, 0) >= DUPLICATE_THRESHOLD:
            s += learned_weights['duplicate'];       reasons.append('duplicate')
        if USERNAME_PATTERN.match(row['用户名称']):
            s += learned_weights['suspicious_user']; reasons.append('suspicious_user')
        key = f"{row['用户ID']}::{row['笔记ID']}"
        if unp_counts.get(key, 0) >= MULTI_DUP_THRESHOLD:
            s += learned_weights['multi_dups'];      reasons.append('multi_dups')
        if row['ttr'] < TTR_THRESHOLD:
            s += learned_weights['low_diversity'];   reasons.append('low_diversity')
        if row['punct_ratio'] < PUNCT_THRESHOLD:
            s += learned_weights['low_punct'];       reasons.append('low_punct')
        return s, ';'.join(reasons)

    scores, reasons = [], []
    for _, row in df.iterrows():
        sc, rs = score_row(row)
        scores.append(sc)
        reasons.append(rs)
    df['可疑度评分'] = scores
    df['可疑原因'] = reasons

    score_arr = df['可疑度评分']
    MED_THR = float(score_arr.quantile(0.80))
    HIGH_THR = float(score_arr.quantile(0.95))
    print(f"动态选阈值：中度疑似 ≥ {MED_THR:.2f}，高疑似 ≥ {HIGH_THR:.2f}")

    plt.figure(figsize=(8, 6))
    plt.hist(score_arr, bins=50, edgecolor='black')
    plt.axvline(HIGH_THR, color='r', linestyle='--', label=f'高疑似阈值 ({HIGH_THR:.2f})')
    plt.axvline(MED_THR, color='orange', linestyle='--', label=f'中度疑似阈值 ({MED_THR:.2f})')
    plt.xlabel('可疑度评分')
    plt.ylabel('频数')
    plt.title('可疑度评分分布与动态阈值')
    plt.legend()
    plt.tight_layout()
    plt.show()

    def label_category(score):
        if score >= HIGH_THR:
            return '高疑似'
        elif score >= MED_THR:
            return '中度疑似'
        else:
            return '正常评论'

    df['初步分类'] = df['可疑度评分'].map(label_category)

    # 补齐 Step 2 所需特征
    df['length'] = df['评论内容'].map(len)
    df['dup_score'] = df['评论内容'].map(lambda x: dup_counts.get(x, 0))
    df['multi_reply_score'] = df[['用户ID','笔记ID']].astype(str).agg('::'.join, axis=1)\
                                   .map(lambda x: unp_counts.get(x, 0))
    df['ad_flag'] = df['评论内容'].map(lambda x: 1 if AD_PATTERN_REGEX.search(x) else 0)

    return df

# =============================
# 主流程执行
# =============================

# 1. 计算动态阈值 & 打印敏感性分析
thresholds = compute_dynamic_thresholds(df)
print_threshold_sensitivity(df, thresholds)

# 2. 训练 Snorkel，获取特征权重
learned_weights = train_snorkel_weights(df, thresholds)

# 3. 打分并动态打标签
df = score_and_label(df, thresholds, learned_weights)

# 4. 保存结果
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print("完成，分类分布：\n", df['初步分类'].value_counts())