# -*- coding: utf-8 -*-

import os
import sys
import io
from pathlib import Path
import importlib
import pandas as pd
from scipy.stats import spearmanr

def get_path(relative_path):
    dir_name = os.path.dirname(__file__)
    return os.path.join(dir_name, relative_path)

module_path = get_path('../submit/src/')
sys.path.append(module_path)

import predictor
import jq_utility as util
from predictor import ScoringService


def main():
    inputs = util.get_inputs(get_path('../inputs'))

    # get model
    ScoringService.get_model(model_path=get_path('../submit/model'))

    # predict
    out = ScoringService.predict(inputs)
    print(len(out))

    # output
    df_pred = pd.read_csv(io.StringIO(out), header=None, names=['date_code', 'pred_high', 'pred_low'])
    df_pred.to_csv(get_path('../outputs/prediction.csv'), index=False)
    print(df_pred.shape)

    # target
    target_col = ['code', 'datetime'] + ScoringService.TARGET_LABELS
    df_target = ScoringService.dfs['stock_labels'][target_col]

    # spearman
    df_pred_w_target = util.add_target(df_target, df_pred, '2020-01-01', '2020-11-30')
    score, spear_high, spear_low = util.calc_spearman_corr(df_pred_w_target)
    print(f'- spearman_score(public)={score:.6f}  ({spear_high:.4f}, {spear_low:.4f})')

    # spearman
    df_pred_w_target = util.add_target(df_target, df_pred, '2020-01-01', '2021-2-26')
    score, spear_high, spear_low = util.calc_spearman_corr(df_pred_w_target)
    print(f'- spearman_score(-2/26)={score:.6f}  ({spear_high:.4f}, {spear_low:.4f})')

    # spearman
    df_pred_w_target = util.add_target(df_target, df_pred, '2021-01-01', '2021-2-26')
    score, spear_high, spear_low = util.calc_spearman_corr(df_pred_w_target)
    print(f'- spearman_score(2021/1/1-2/26)={score:.6f}  ({spear_high:.4f}, {spear_low:.4f})')


if __name__ == "__main__":
    main()