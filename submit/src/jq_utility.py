# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import spearmanr

def get_inputs(dataset_dir):
    """
    Args:
        dataset_dir (str)  : path to dataset directory
    Returns:
        dict[str]: path to dataset files
    """
    inputs = {
        "stock_list": f"{dataset_dir}/stock_list.csv",
        "stock_price": f"{dataset_dir}/stock_price.csv",
        "stock_fin": f"{dataset_dir}/stock_fin.csv",
        "stock_labels": f"{dataset_dir}/stock_labels.csv",
    }
    return inputs


def add_target(df_target, df_pred, start, end):
    target_cp = df_target[(df_target['datetime'] >= start)&(df_target['datetime'] <= end)].copy()
    target_cp = target_cp[target_cp['label_high_20'].notnull()]
    target_cp['date_code'] = target_cp['datetime'].astype(str) + '-' + target_cp['code'].astype(str)
    df_merge = pd.merge(df_pred, target_cp, on='date_code', how='inner')
    print(df_merge.shape)
    return df_merge

def spearman(data, highlow):
    xlabel = f'label_{highlow}_20'
    ylabel = f'pred_{highlow}'
    return spearmanr(data[xlabel], data[ylabel])[0]

def calc_spearman_corr(df_merge):
    df_merge['rank_high'] = df_merge['label_high_20'].rank()
    df_merge['rank_low'] = df_merge['label_low_20'].rank()
    df_merge['rank_pred_high'] = df_merge['pred_high'].rank()
    df_merge['rank_pred_low'] = df_merge['pred_low'].rank()

    spear_high = spearman(df_merge, 'high')
    spear_low = spearman(df_merge, 'low')
    score = (spear_high-1)**2 + (spear_low-1)**2
    return score, spear_high, spear_low