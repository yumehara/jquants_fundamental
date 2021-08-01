# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import Pool
from catboost import CatBoostRegressor
import warnings
warnings.simplefilter('ignore', UserWarning)

ALGO = ['lgbm', 'catboost']

lgbm_lr = 0.1

lgbm_param = {
    'label_high_20': {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'seed': 0,
        'learning_rate': lgbm_lr,
        'verbosity': -1,
        'feature_pre_filter': False,
        'lambda_l1': 5.332187932922965,
        'lambda_l2': 0.001349205704669796,
        'num_leaves': 121,
        'feature_fraction': 0.5,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'min_child_samples': 10,
    },
    'label_low_20': {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'seed': 0,
        'learning_rate': lgbm_lr,
        'verbosity': -1,
        'feature_pre_filter': False,
        'lambda_l1': 4.584123770031804,
        'lambda_l2': 0.28379273481211037,
        'num_leaves': 127,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.9675232152701854,
        'bagging_freq': 7,
        'min_child_samples': 100,
    }
}

def lgbm_train(X, y, k_fold, cat_list, label):
    params = lgbm_param[label]
    rmse_list = []
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)
    
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=0)
    cv_split = cv.split(X)

    for train_index, valid_index in cv_split:

        x_train, x_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        lgbm_train = lgbm.Dataset(x_train, y_train, categorical_feature=cat_list)
        lgbm_eval = lgbm.Dataset(x_valid, y_valid, reference=lgbm_train, categorical_feature=cat_list)
        lgbm_model = lgbm.train(params, 
                                lgbm_train, 
                                valid_sets=lgbm_eval,
                                num_boost_round=10000,
                                early_stopping_rounds=100,
                                categorical_feature=cat_list,
                                verbose_eval=-1)
        y_pred = lgbm_model.predict(x_valid, num_iteration=lgbm_model.best_iteration)
        oof_pred[valid_index] = y_pred

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_list.append(rmse)
        models.append(lgbm_model)

    score = np.sqrt(mean_squared_error(y, oof_pred))
    return models, oof_pred, score

cat_params = {
    'label_high_20': {
        'eval_metric' :'RMSE', 
        'random_seed' :0,
        'num_boost_round': 10000,
        'depth': 9, 
        'learning_rate': 0.06928902194279778, 
        'random_strength': 33, 
        'bagging_temperature': 0.011872992424228708, 
        'od_type': 'IncToDec', 
        'od_wait': 20
    },
    'label_low_20': {
        'eval_metric' :'RMSE', 
        'random_seed' :0,
        'num_boost_round': 10000,
        'depth': 8, 
        'learning_rate': 0.0744631476349798, 
        'random_strength': 71, 
        'bagging_temperature': 0.4793066522800233, 
        'od_type': 'IncToDec', 
        'od_wait': 20
    }
}

def catboost_train(X, y, k_fold, cat_list, label):
    params = cat_params[label]
    rmse_list = []
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)

    cv = KFold(n_splits=k_fold, shuffle=True, random_state=100)
    cv_split = cv.split(X)

    for idx_train, idx_valid in cv_split:

        x_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        x_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
        
        train_pool = Pool(data=x_train, label=y_train, cat_features=cat_list)
        validate_pool = Pool(data=x_valid, label=y_valid, cat_features=cat_list)
        cat_model = CatBoostRegressor(**params)
        cat_model.fit(train_pool, eval_set=validate_pool, 
                                  verbose=100, 
                                  use_best_model=True,
                                  early_stopping_rounds=100)
        y_pred = cat_model.predict(validate_pool)
        oof_pred[idx_valid] = y_pred

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_list.append(rmse)
        models.append(cat_model)

    score = np.sqrt(mean_squared_error(y, oof_pred))
    return models, oof_pred, score

def lgbm_predict(model, feature):
    pred = model.predict(feature, num_iteration = model.best_iteration)
    return pred

def catboost_predict(model, feature, cat_feat):
    test_pool = Pool(data=feature, cat_features=cat_feat)
    pred = model.predict(test_pool)
    return pred

def train(algo, X, y, k_fold, cat_list, label):
    if algo == 'lgbm':
        return lgbm_train(X, y, k_fold, cat_list, label)
    elif algo == 'catboost':
        return catboost_train(X, y, k_fold, cat_list, label)

def predict(algo, model, feature, cat_list):
    if algo == 'lgbm':
        return lgbm_predict(model, feature)
    elif algo == 'catboost':
        return catboost_predict(model, feature, cat_list)
