# -*- coding: utf-8 -*-
import io
import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import preprocess as prep
import jq_ml


class ScoringService(object):
    # 訓練期間開始日
    TRAIN_START = '2016-06-01'
    # 訓練期間終了日
    TRAIN_END = '2019-12-01'
    # テスト期間開始日
    TEST_START = '2020-01-01'
    # 目的変数
    TARGET_LABELS = ['label_high_20', 'label_low_20']

    KFOLD = 5
    cat_list = ['code', '33 Sector(Code)', 'Size Code (New Index Series)', 'Section/Products']

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None
    # debug用
    features = None
    pred_feats = None


    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            if k == "stock_price":
                prep.prep_stock_price(cls.dfs[k])
            elif k == 'stock_fin':
                prep.prep_stock_fin(cls.dfs[k])
            elif k == 'stock_labels':
                prep.prep_common_data(cls.dfs[k])
            elif k == 'stock_list':
                prep.prep_stock_list(cls.dfs[k])
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"]
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True]["code"].values
        return cls.codes

    @classmethod
    def create_model(cls, dfs, feature, label, end_dt, model_path):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        print('create_model', label)

        # stock_labelデータを読み込み
        stock_labels = dfs["stock_labels"][['code', 'datetime', label]].dropna()

        merge = pd.merge(feature, stock_labels, on=['datetime','code'], how='inner')
        trains = merge[merge['datetime'] <= end_dt]
        train_X = trains.drop(columns=['datetime', label])
        train_y = trains[label]

        algo_score = {}
        # モデル作成
        for algo in jq_ml.ALGO:
            models, _, score = jq_ml.train(algo, train_X, train_y, cls.KFOLD, cls.cat_list, label)
            algo_score[algo] = score
            cls.save_model(algo, models, label, model_path)

        print('- train_X', train_X.shape)
        return algo_score

    @classmethod
    def save_model(cls, algo, models, label, model_path):
        for i, model in enumerate(models):
            filename = f'{algo}_{label}_{i}.pkl'
            print('save_model:', filename)
            with open(os.path.join(model_path, filename), 'wb') as f:
                pickle.dump(model, f)

    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        labels = cls.TARGET_LABELS

        # try:
        for label in labels:
            algo_model = {}
            for algo in jq_ml.ALGO:
                model_list = []
                for i in range(cls.KFOLD):
                    filename = f'{algo}_{label}_{i}.pkl'
                    print('get_model:', filename)
                    m = os.path.join(model_path, filename)
                    with open(m, "rb") as f:
                        model_list.append(pickle.load(f))
                algo_model[algo] = model_list
            
            cls.models[label] = algo_model
        return True
        # except Exception as e:
        #     print(e)
        #     return False

    @classmethod
    def train_and_save_model(cls, inputs, model_path, start_dt=TRAIN_START, end_dt=TRAIN_END, feat_filepath=None, is_read_feat=False):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        codes = cls.codes
        labels = cls.TARGET_LABELS

        if not is_read_feat:
            # 特徴量生成
            feature = pd.DataFrame()
            for code in tqdm(codes):
                tmp = prep.get_all_features(dfs=cls.dfs, code=code, start_dt=start_dt)
                feature = pd.concat([feature, tmp])

            if feat_filepath is not None:
                feature.to_pickle(feat_filepath)
        else:
            # 特徴量読み込み
            if feat_filepath is not None:
                feature = pd.read_pickle(feat_filepath)
        cls.features = feature

        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)

        score_dict = {}
        for label in labels:
            print(label)
            algo_score = cls.create_model(dfs=cls.dfs, feature=feature, label=label, end_dt=end_dt, model_path=model_path)
            score_dict[label] = algo_score
        
        print(f'- {start_dt} - {end_dt}')
        print('- LGBM_SCORE={:.4f}, {:.4f}'.format(score_dict['label_high_20']['lgbm'], score_dict['label_low_20']['lgbm']))
        print('- CATB_SCORE={:.4f}, {:.4f}'.format(score_dict['label_high_20']['catboost'], score_dict['label_low_20']['catboost']))

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        feats = pd.DataFrame()
        for code in tqdm(codes):
            tmp = prep.get_all_features(dfs=cls.dfs, code=code, start_dt=start_dt)
            feats = pd.concat([feats, tmp])

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(str)

        # 出力対象列を定義
        output_columns = ["code"]

        print('- test_X', feats.shape)
        cls.pred_feats = feats

        lgbm_rate = 0.1
        # 目的変数毎に予測
        for label in labels:
            algo_pred = {}
            for algo in jq_ml.ALGO:
                model_list = cls.models[label][algo]
                pred_list = []
                for model in model_list:
                    pred = jq_ml.predict(algo, model, feats, cls.cat_list)
                    pred_list.append(pred)
                
                algo_pred[algo] = np.mean(pred_list, axis=0)
                
            df[label] = algo_pred['lgbm']*lgbm_rate + algo_pred['catboost']*(1-lgbm_rate)
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()