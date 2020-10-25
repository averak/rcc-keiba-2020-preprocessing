# 競馬 AI 班 前処理

[![Twitter](https://img.shields.io/badge/Twitter-競馬AI班-blue?style=flat-square&logo=twitter)](https://twitter.com/search?q=%23rcc_keiba)

RCC 2020 年度プロジェクト活動

## 概要

非整形データに前処理を加える練習です。

[大津市の天気](https://drive.google.com/u/0/uc?id=1bncXn4z5ZsgP_6fW3i-HjQdc2WxtEsty&export=download)を使って，降水の有無を予測しましょう。

## 実行環境

- OS：問わない
- Python ~> 3.8

## インストールと学習

### データの準備

[weather.csv](https://drive.google.com/u/0/uc?id=1bncXn4z5ZsgP_6fW3i-HjQdc2WxtEsty&export=download)をダウンロードする

### インストール

```sh
$ git clone <this repo>
$ cd <this repo>

$ pip install -U pip
$ pip install -r requirements.txt
```

### 主成分の寄与率を計算
```sh
$ python pca.py
```

### 学習

```sh
$ ./train.py
```

## 作成者

- [Averak](https://github.com/averak)
