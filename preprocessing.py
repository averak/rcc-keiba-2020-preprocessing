import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def is_rain(x):
    return int(x == '有')

def wind_direction(x):
    azimuth = {
        '東': 90,
        '南': 180,
        '西': 270,
        '北': 360,
    }
    return sum([azimuth[c] for c in x]) / len(x)

# csv読み込み
df = pd.read_csv('weather.csv')
df = df.drop('date', axis=1)

# カテゴリデータの処理
df['is_rain'] = df['is_rain'].map(is_rain)
df['wind_direction'] = df['wind_direction'].map(wind_direction)

# 標準化
scaler = StandardScaler()
scaler.fit(df[df.columns[:8]])
df[df.columns[:8]] = pd.DataFrame(scaler.transform(
    df[df.columns[:8]]), columns=df.columns[:8])

# ==========================================
# 今回のコード（PCA）
from sklearn.decomposition import PCA

# データセットの準備
feature = np.array(df)
X = feature[:, :8]
Y = feature[:, 8]

# 主成分分析の実行
pca = PCA(n_components=4)
pca.fit(X)

# 次元削減
X = pca.transform(X)
# ==========================================
