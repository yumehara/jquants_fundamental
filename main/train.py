# -*- coding: utf-8 -*-

import os
import sys
import io
from pathlib import Path
import importlib
import pandas as pd
import glob

def get_path(relative_path):
    dir_name = os.path.dirname(__file__)
    return os.path.join(dir_name, relative_path)

module_path = get_path('../submit/src/')
sys.path.append(module_path)

import predictor
import jq_utility as util
from predictor import ScoringService


def main():

    for file in glob.glob(get_path('../submit/model/*.pkl')):
        print("remove：{0}".format(file))
        os.remove(file)

    inputs = util.get_inputs(get_path('../inputs'))

    # 訓練期間開始日
    TRAIN_START = '2016-06-01'
    # 訓練期間終了日
    TRAIN_END = '2021-02-26'

    # train model
    ScoringService.train_and_save_model(
        inputs, model_path=get_path('../submit/model'), start_dt=TRAIN_START, end_dt=TRAIN_END,
        feat_filepath=get_path('../outputs/feature.pkl'), is_read_feat=False)


if __name__ == "__main__":
    main()