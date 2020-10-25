import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

# csvデータを読み込む
# 必要なデータのみ抜き出す
# 説明変数らを標準化(正規化)する
# 標準化したデータの分散共分散行列を計算
# ラグランジュの乗数法から相関行列による固有値固有ベクトルの式

# 標準化する際のカテゴリデータの処理関数
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

# csvデータを読み込み
df = pd.read_csv('weather.csv')
df = df.drop('date', axis=1)
#print(df)

# カテゴリデータの処理
df['is_rain'] = df['is_rain'].map(is_rain)
df['wind_direction'] = df['wind_direction'].map(wind_direction)

# 取得したデータの確認
print('=== データ項目は以下を選択 ===')
print(df.columns)

# 説明変数らを標準化する
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# 主成分分析の実行
pca = PCA()
pca.fit(df)

# 次元削減
feature = pca.transform(df)

# 主成分得点を表形式に成形する
pca_graph = pd.DataFrame(feature, columns=['PC{}'.format(x + 1)
                                           for x in df.columns]).head()

# 実行結果を出力する
print('=== それぞれの主成分得点の表 ===')
print(pca_graph)
#print('=== 主成分分析をした結果得た特徴の値 ===')
#print(feature[:, 0])
print('=== 寄与率 ===')
print(pca.explained_variance_ratio_)


cnt = 0
sum_ratio = 0
for x in pca.explained_variance_ratio_:
    if sum_ratio < 0.85:
        cnt += 1
        sum_ratio += x
    else:
        break

print('%d次元を%d次元まで圧縮可能' % (df.shape[1],cnt))
print('全体の分散の%3.1f%%を確保できる' % (sum_ratio * 100))
