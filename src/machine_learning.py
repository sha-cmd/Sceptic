import os
import os.path as path
import pandas as pd
import numpy as np
from glob import glob
from surprise import accuracy
from objects.BookLib import BookLib
from surprise.model_selection import train_test_split
from surprise import prediction_algorithms

book_library = BookLib()
data = book_library()
train_set, test_set = train_test_split(data, test_size=.25)

met = {'rmse': accuracy.rmse,
       'mae': accuracy.mae,
       'mse': accuracy.mse,
       'fcp': accuracy.fcp}
pred = {'NormalPred': ['metrics/normalPred', prediction_algorithms.random_pred.NormalPredictor],
        'baseLineALS': ['metrics/baseLineALS', prediction_algorithms.baseline_only.BaselineOnly],
        'baseLineSGD': ['metrics/baseLineSGD', prediction_algorithms.baseline_only.BaselineOnly],
        'KNNBasicALS': ['metrics/knnBasicALS', prediction_algorithms.knns.KNNBasic],
        'KNNBasicSimsCos': ['metrics/knnBasicSims', prediction_algorithms.knns.KNNBasic],
        'KNNBasicSimsMsd': ['metrics/knnBasicSims', prediction_algorithms.knns.KNNBasic],
        'KNNBasicPearson': ['metrics/knnBasicNoShrink', prediction_algorithms.knns.KNNBasic],
        'SVD': ['metrics/SVD', prediction_algorithms.matrix_factorization.SVD],
        'SVDpp': ['metrics/SVDpp', prediction_algorithms.matrix_factorization.SVDpp],
        'NMF': ['metrics/NMF', prediction_algorithms.matrix_factorization.NMF]
        }

for algo_name in pred.keys():
    dir_list = pred[algo_name][0].split('/')
    print(algo_name)
    algo = None
    # We'll use the famous SVD algorithm.
    if algo_name == 'baseLineALS':
        bsl_options = {'method': 'als',
                       'n_epochs': 5,
                       'reg_u': 12,
                       'reg_i': 5
                       }
        algo = pred[algo_name][1](bsl_options=bsl_options)
    elif algo_name == 'baseLineSGD':
        bsl_options = {'method': 'sgd',
                       'learning_rate': .00005,
                       }
        algo = pred[algo_name][1](bsl_options=bsl_options)
    elif algo_name == 'KNNBasicALS':
        bsl_options = {'method': 'als',
                       'n_epochs': 20,
                       }
        sim_options = {'name': 'pearson_baseline'}
        algo = pred[algo_name][1](bsl_options=bsl_options, sim_options=sim_options)
    elif algo_name == 'KNNBasicSimsCos':
        sim_options = {'name': 'cosine',
                       'user_based': False  # compute  similarities between items
                       }
        algo = pred[algo_name][1](sim_options=sim_options)
    elif algo_name == 'KNNBasicSimsMsd':
        sim_options = {'name': 'msd',
                       'user_based': False  # compute  similarities between items
                       }
        algo = pred[algo_name][1](sim_options=sim_options)
    elif algo_name == 'KNNBasicPearson':
        sim_options = {'name': 'pearson'
                       }
        algo = pred[algo_name][1](sim_options=sim_options)
    else:
        algo = pred[algo_name][1]()

    # Run 5-fold cross-validation and print results
    algo.fit(train_set)
    predictions = algo.test(test_set)

    for it, dirs in enumerate(dir_list):
        if not path.isdir("/".join(dir_list[:it + 1])):
            os.makedirs("/".join(dir_list[:it + 1]))

    metric_list = ['rmse', 'mae', 'mse', 'fcp']
    for met_name in metric_list:
        with open(pred[algo_name][0] + '/' + met_name + '.txt', 'w') as f:
            score = met[met_name](predictions)
            f.write(str(score))
            pd.DataFrame([[score]], columns=[met_name]) \
                .to_csv(pred[algo_name][0] + "/" + met_name + ".tsv", index_label='index', sep='\t')
# Écriture de la synthèse
liste = glob('metrics/**/*.tsv')
rmse_list = {'rmse': [pd.read_csv(x, sep='\t', index_col='index').values[0][0] for x in liste if
             x.split('/')[-1][:4] == 'rmse']}
mse_list = {'mse': [pd.read_csv(x, sep='\t', index_col='index').values[0][0] for x in liste if
             x.split('/')[-1][:3] == 'mse']}
mae_list = {'mae': [pd.read_csv(x, sep='\t', index_col='index').values[0][0] for x in liste if
             x.split('/')[-1][:3] == 'mae']}
fcp_list = {'fcp': [pd.read_csv(x, sep='\t', index_col='index').values[0][0] for x in liste if
             x.split('/')[-1][:3] == 'fcp']}
name_list = {'name': list(set([x.split('/')[1] for x in liste]))}
data = {}
data.update(name_list)
data.update(rmse_list)
data.update(mse_list)
data.update(mae_list)
data.update(fcp_list)
pd.DataFrame.from_dict(data).to_csv('metrics/synthese.csv', index_label='index', sep=';')