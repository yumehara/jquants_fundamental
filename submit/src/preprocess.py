# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def prep_common_data(input):
    input['datetime'] = pd.to_datetime(input['base_date'])
    input.rename(columns={'Local Code':'code'}, inplace=True)

def prep_stock_fin(input):
    prep_common_data(input)
    peportyype_dict = {
        'Q1':1,
        'Q2':2,
        'Q3':3,
        'Annual':4,
    }
    input['ReportType_Int'] = input['Result_FinancialStatement ReportType'].map(peportyype_dict)
    input.sort_values('datetime', inplace=True)
    input.reset_index(drop=True, inplace=True)

# ファンダメンタル情報
def get_fin_feature(stock_fin, code, start_dt, n):
    # 特定の銘柄コードのデータに絞る
    fin_data = stock_fin[stock_fin['code'] == code]
    # 特徴量の生成対象期間を指定
    prep_start = pd.Timestamp(start_dt) - pd.offsets.BDay(n)
    fin = fin_data[fin_data['datetime'] >= prep_start].copy()
    # 四半期ごとの修正版
    fin['version_number'] = fin.groupby(
        ['Result_FinancialStatement FiscalYear', 'Result_FinancialStatement ReportType']
    )['datetime'].rank()
    # 最後の修正のみを残す
    fin['version_number_inv'] = fin.groupby(
        ['Result_FinancialStatement FiscalYear', 'Result_FinancialStatement ReportType']
    )['datetime'].rank(ascending=False)
    fin = fin[fin['version_number_inv']==1].drop(columns=['version_number_inv'])
    fin_len = len(fin)

    # 前年度との差分
    # fin = pd.concat([fin.reset_index(drop=True), diff_1year_ago(fin)], axis=1)
    # 前Qとの差分
    fin = pd.concat([fin.reset_index(drop=True), diff_1quater_ago(fin)], axis=1)
    assert len(fin) == fin_len

    # 不要なカラムを削除
    fin.drop(columns=[
        'Forecast_FinancialStatement FiscalYear',
        'Result_Dividend FiscalYear',
        'Forecast_Dividend FiscalYear'
    ], inplace=True)

    # 穴埋め
    def fill_f(fin, col):
        fin[col] = fin[col].fillna(method='ffill')
    
    for col in [
        'Result_FinancialStatement CashFlowsFromOperatingActivities',
        'Result_FinancialStatement CashFlowsFromFinancingActivities',
        'Result_FinancialStatement CashFlowsFromInvestingActivities',]:
        fill_f(fin, col)

    # fin_dataのnp.float64のデータのみを取得
    fin = fin.select_dtypes(include=['float64','datetime64[ns]'])
    return fin

diff_target_columns = [
    'Result_FinancialStatement NetSales', # 売上高
    'Result_FinancialStatement OperatingIncome', # 営業利益
    'Result_FinancialStatement OrdinaryIncome', # 経常利益
    'Result_FinancialStatement NetIncome', # 当期純利益
    'Result_FinancialStatement TotalAssets', # 総資産
    'Result_FinancialStatement NetAssets', # 純資産
    'Result_Dividend QuarterlyDividendPerShare', # 配当情報: 一株当たり四半期配当金
    'Result_Dividend AnnualDividendPerShare', # 配当情報: 一株当たり年間配当金累計
]

# 前年度との差分
def diff_1year_ago(df_fin):
    fin = df_fin.copy()
    past_fin = df_fin.copy()
    suffix = '_past1'
    past_fin = past_fin.add_suffix(suffix)
    past_fin.rename(columns={
        'code_past1':'code', 
        'Result_FinancialStatement ReportType_past1':'Result_FinancialStatement ReportType'
    }, inplace=True)

    key_columns = ['code', 'Result_FinancialStatement FiscalYear_past1', 'Result_FinancialStatement ReportType']
    past_columns = list(map(lambda x: x+suffix, diff_target_columns))
    past_fin = past_fin[key_columns+past_columns]
    fin['Result_FinancialStatement FiscalYear'+suffix] = fin['Result_FinancialStatement FiscalYear'] - 1
    fin = pd.merge(fin, past_fin, on=key_columns, how='left')

    prefix = 'pct_year_'
    for col in diff_target_columns:
        fin[prefix+col] = (fin[col] - fin[col+suffix])/fin[col+suffix]
    
    return_columns = list(map(lambda x: prefix+x, diff_target_columns))
    return fin[return_columns]

# 前Qとの差分
def diff_1quater_ago(df_fin):
    fin = df_fin.copy()
    prefix = 'pct_preQ_'
    quater = 'ReportType_Int'
    diff = fin[diff_target_columns].pct_change(1).add_prefix(prefix)
    fin = pd.concat([fin[[quater]], diff], axis=1)

    for col in diff_target_columns:
        target = prefix + col
        fin.loc[fin[quater] == 1, target] = np.NaN
    
    return fin.drop(columns=[quater]).reset_index(drop=True)
    

# 株価情報
def prep_stock_price(stock_price):
    stock_price['datetime'] = pd.to_datetime(stock_price['EndOfDayQuote Date'])
    stock_price.rename(columns={'Local Code':'code'}, inplace=True)

# 株価情報
def get_price_feature(price, code, start_dt, n):
    # 特定の銘柄コードのデータに絞る
    price_data = price[price['code'] == code]
    # 終値のみに絞る
    feats = price_data[[
        'datetime', 'EndOfDayQuote ExchangeOfficialClose',
        'EndOfDayQuote High', 'EndOfDayQuote Low',
        'EndOfDayQuote ChangeFromPreviousClose',
        'EndOfDayQuote PercentChangeFromPreviousClose',
    ]]
    # 特徴量の生成対象期間を指定
    prep_start = pd.Timestamp(start_dt) - pd.offsets.BDay(n)
    feats = feats[feats['datetime'] >= prep_start].copy()
    feats.rename(columns={
        'EndOfDayQuote ExchangeOfficialClose':'Close',  # 取引所公式終値
        'EndOfDayQuote High':'High',                    # 高値
        'EndOfDayQuote Low':'Low',                      # 安値
        'EndOfDayQuote ChangeFromPreviousClose':'cfpc', # 騰落幅 前回終値との価格差
        'EndOfDayQuote PercentChangeFromPreviousClose':'pcfpc', # 騰落率 前回終値からの上昇/下落率
    }, inplace=True)

    for d in [20, 40, 60]:
        # 終値のd営業日リターン
        feats[f'return_{d}day'] = feats['Close'].pct_change(d)
        # 終値の20営業日ボラティリティ
        feats[f'volatility_{d}day'] = np.log(feats['Close']).diff().rolling(d).std()
        # 終値と20営業日の単純移動平均線の乖離
        feats[f'MA_gap_{d}day'] = feats['Close']/feats['Close'].rolling(d).mean()
        # (最大-最小)/標準偏差
        feats[f'max_min_{d}day'] = (feats['High'].rolling(d).max() - feats['Low'].rolling(d).min())/feats['Close'].rolling(d).std()
        # (最大-最小)の平均/平均
        feats[f'max_min_mean_{d}day'] = (feats['High']-feats['Low']).rolling(d).mean()/feats['Close'].rolling(d).mean()
        # 騰落率の平均
        feats[f'PercentChange_mean_{d}day'] = feats['pcfpc'].rolling(d).mean()
        # 騰落率の標準偏差
        feats[f'PercentChange_std_{d}day'] = feats['pcfpc'].rolling(d).std()
        # 騰落幅の標準偏差/平均
        feats[f'Change_std_mean_{d}day'] = feats['cfpc'].rolling(d).std()/feats['cfpc'].rolling(d).mean()
        # RSI
        feats[f'RSI_{d}day'] = calc_RSI(feats['cfpc'], d)
    
    # MACD
    feats['MACD'] = calc_MACD(feats['Close'])
    
    return feats.drop(columns=['High', 'Low', 'cfpc', 'pcfpc'])

# RSI(相対力指数)
def calc_RSI(close_diff, period):
    up, down = close_diff.copy(), close_diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    up_sma = up.rolling(period).mean()
    down_sma = down.abs().rolling(period).mean()
    return (up_sma/(up_sma+down_sma))

# MACD(Moving Average Convergence Divergence)
def calc_MACD(close_price):
    MACD = close_price.ewm(span=12).mean() - close_price.ewm(span=26).mean()
    signal = MACD.ewm(span=9).mean()
    return (MACD - signal)/MACD

# 銘柄情報の前処理
def prep_stock_list(stocklist):
    stocklist['Size Code (New Index Series)'] = stocklist['Size Code (New Index Series)'].replace({'-':0}).astype(int)
    section_dict = {
        'First Section (Domestic)':0,
        'JASDAQ(Standard / Domestic)':1,
        'Second Section(Domestic)':2,
        'Mothers (Domestic)':3,
        'JASDAQ(Growth/Domestic)':4,
    }
    stocklist['Section/Products'] = stocklist['Section/Products'].map(section_dict)
    stocklist['IssuedShare_bin'] = pd.qcut(stocklist['IssuedShareEquityQuote IssuedShare'], 100, labels=False)
    stocklist.rename(columns={'Local Code':'code'}, inplace=True)

# 銘柄情報
def get_comp_feature(stocklist, code):
    # 特定の銘柄コードのデータに絞る
    feats = stocklist[stocklist['code'] == code]
    return feats[[
        'code', '33 Sector(Code)',
        'Size Code (New Index Series)', 'Section/Products',
        'IssuedShareEquityQuote IssuedShare', 'IssuedShare_bin'
    ]]

# 全ての特徴量を集約
def get_all_features(dfs, code, start_dt):
    """
    Args:
        dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
        code (int)  : A local code for a listed company
        start_dt (str): specify date range
    Returns:
        feature DataFrame (pd.DataFrame)
    """
    # 特徴量の作成には過去60営業日のデータを使用しているため、
    # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
    n = 90

    # ファンダメンタル情報(stock_fin)の特徴量生成
    fin_feats = get_fin_feature(dfs['stock_fin'], code, start_dt, n)

    # 株価情報(stock_price)の特徴量生成
    price_feats = get_price_feature(dfs['stock_price'], code, start_dt, n)

    feats = pd.merge(price_feats, fin_feats, on='datetime', how='inner')
    feats['code'] = code

    # 銘柄情報(stock_list)の特徴量生成
    comp_feats = get_comp_feature(dfs['stock_list'], code)
    feats = pd.merge(feats, comp_feats, on='code', how='left')

    # カテゴリの特徴量をint型に
    cat_feat = ['33 Sector(Code)', 'Size Code (New Index Series)', 'Section/Products']
    for cat in cat_feat:
        feats[cat] = feats[cat].astype(int)

    # ROE（自己資本利益率）= 当期純利益 / 自己資本（純資産）
    feats['ROE'] = feats['Result_FinancialStatement NetIncome']/feats['Result_FinancialStatement NetAssets']
    # ROA（総資産利益率）= 当期純利益 / 総資産
    feats['ROA'] = feats['Result_FinancialStatement NetIncome']/feats['Result_FinancialStatement TotalAssets']
    # EPS（1株当たり利益）= 当期純利益 / 発行済株式総数
    feats['EPS'] = feats['Result_FinancialStatement NetIncome']/feats['IssuedShareEquityQuote IssuedShare']
    # PER（株価収益率）= 株価 / 1株当たり利益（EPS）予測値
    feats['PER'] = feats['Close']/(feats['Forecast_FinancialStatement NetIncome']/feats['IssuedShareEquityQuote IssuedShare'])
    # BPS（1株あたり純資産）= 純資産 / 発行済株式総数
    feats['BPS'] = feats['Result_FinancialStatement NetAssets']/feats['IssuedShareEquityQuote IssuedShare']
    # PBR（株価純資産倍率）= 株価 / BPS（1株あたり純資産）
    feats['PBR'] = feats['Close']/feats['BPS']
    # 配当性向 = 配当金支払総額 / 当期純利益 
    feats['Dividend_payout_ratio'] = feats['Result_Dividend AnnualDividendPerShare']/feats['Result_FinancialStatement NetIncome']

    feats.drop(columns=['Close'], inplace=True)

    # 欠損値処理
    feats = feats.replace([np.inf, -np.inf], 0)

    # 生成対象日以降の特徴量に絞る
    feats = feats[feats['datetime'] >= pd.Timestamp(start_dt)]
    return feats.set_index('datetime')

