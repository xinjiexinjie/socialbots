import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('/Users/xinjie/PycharmProjects/socialbots/data.csv')

# 2. 删除“评论图片”列
df = df.drop(columns=['评论图片'])

# 删除列名以 "Unnamed" 开头的列（通常是空白冗余列）
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

# 3. 把空字符串也当作缺失值处理
df['评论内容'] = df['评论内容'].replace('', np.nan)

# 4. 删除“评论内容”为空的行
df = df.dropna(subset=['评论内容'])

# 5. 根据“评论ID”去重，保留首次出现
df = df.drop_duplicates(subset=['评论ID'], keep='first')

# 6. （可选）重置索引
df = df.reset_index(drop=True)

# 7. 保存清洗后的数据
out_path = '/Users/xinjie/PycharmProjects/socialbots/data_cleaned.csv'
df.to_csv(out_path, index=False)

print(f'清洗完成，剩余行数：{len(df)}，结果已保存到 {out_path}')