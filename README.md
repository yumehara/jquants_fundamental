# 日本取引所グループ ファンダメンタルズ分析チャレンジ
## 4th Place Solution (by y.ume)

### 環境
#### OS
- Ubuntu 18.04.5 LTS

#### 言語
- Python 3.7.3

#### ライブラリ
- anaconda3:2019.03
- lightgbm 3.1.1
- catboost 0.24.4

#### GPU
- 使用していない

### ソースコード概要
- submit/src
  - predictor.py: ScoringServiceクラスの定義
  - preprocess.py: 特徴量生成に関する関数
  - jq_ml.py: 機械学習の学習と予測に関する関数
  - jq_utility.py: get_inputs関数などを定義
- submit/model
  - 学習済みのモデルを保存（lightgbm 10個、catboost 10個）
- main
  - main.py: エントリポイント(trainとpredictを呼び出すだけ)
  - train.py: ScoringServiceクラスのtrain_and_save_modelを呼び出す（提出したモデルの学習期間は、2016-06-01から2021-02-26）
  - predict.py: ScoringServiceクラスのget_modelの後、predictを呼び出す。出力結果から期間を変えて順位相関係数を求めている
- inputs
  - 入力のCSVファイルを配置する（配置したファイル名に応じて、jq_utility.pyのget_inputs関数を変える）
	
### 処理の流れ
- 基本的な処理はチュートリアルと同じ

#### データ加工及びモデル学習
- ScoringServiceクラスのtrain_and_save_model関数で行う
- train_and_save_model関数
  - 特徴量生成は、preprocess.pyのget_all_features関数で行っている
  - 生成した特徴量を元に、ScoringServiceクラスのcreate_model関数内でモデルを作成する
  - 機械学習部分は、jq_ml.pyにまとめている

#### 予測
- ScoringServiceクラスのget_model関数を呼び出した後、ScoringServiceクラスのpredict関数で予測結果を出力する
- predict関数
  - preprocess.pyのget_all_features関数で、テストデータに対する特徴量生成を行い、jq_ml.pyのpredict関数で実際の予測計算を行っている

### 実行手順
- ```docker-compose up --build```
をすることで、dockerコンテナ上で、trainからpredictまでを一貫して行う。
- 学習に使うデータについては、J-Quants APIで2021-03-26までのデータを取得して、元のデータにマージしたファイルを使用している。
- 実際に学習に使用した期間は、2016-06-01から2021-02-26まで

### 特徴量
- 基本的には、株価の分析でよく使われているテクニカルおよびファンダメンタルの指標を網羅するようにした。
- テクニカル
    - 20,40,60営業日のそれぞれについて、リターン、ボラティリティ、移動平均との乖離率、(最大-最小)/標準偏差、騰落率の平均、騰落率の標準偏差、騰落幅の標準偏差/平均、RSI
    - MACD
- ファンダメンタル
    - ROE, ROA, EPS, PER, BPS, PBR, 配当性向、四半期ごとの修正回数、
    - 前四半期との差分（売上高、営業利益、経常利益、当期純利益、総資産、純資産、一株当たり四半期配当金、一株当たり年間配当金累計）

### モデリング
- 最高値と最安値ごとにモデルを分け、それぞれ5FOLDでlightgbmとcatboostのモデルを作成したため、全部で20個のモデルを作成した。
- 最終的に、直近のテストデータの評価値が最も高かった比率でアンサンブルを行って提出データを作成した。
- 学習時は時系列データとしての考慮は特にしておらず、厳密に言えばリークが発生している可能性はあるが、面倒だったので何も対策は行わなかった。



