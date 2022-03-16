import os
import os.path as path
import pandas as pd
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

for algo in pred.keys():
    dir_list = pred[algo][0].split('/')
    algo = None
    print(algo)
    # We'll use the famous SVD algorithm.
    if algo == 'baseLineALS':
        bsl_options = {'method': 'als',
                       'n_epochs': 5,
                       'reg_u': 12,
                       'reg_i': 5
                       }
        algo = pred[algo][1](bsl_options=bsl_options)
    elif algo == 'baseLineSGD':
        bsl_options = {'method': 'sgd',
                       'learning_rate': .00005,
                       }
        algo = pred[algo][1](bsl_options=bsl_options)
    elif algo == 'KNNBasicALS':
        bsl_options = {'method': 'als',
                       'n_epochs': 20,
                       }
        sim_options = {'name': 'pearson_baseline'}
        algo = pred[algo][1](bsl_options=bsl_options, sim_options=sim_options)
    elif algo == 'KNNBasicSimsCos':
        sim_options = {'name': 'cosine',
                       'user_based': False  # compute  similarities between items
                       }
        algo = pred[algo][1](sim_options=sim_options)
    elif algo == 'KNNBasicSimsMsd':
        sim_options = {'name': 'msd',
                       'user_based': False  # compute  similarities between items
                       }
        algo = pred[algo][1](sim_options=sim_options)
    elif algo == 'KNNBasicPearson':
        sim_options = {'name': 'pearson'
                       }
        algo = pred[algo][1](sim_options=sim_options)
    else:
        algo = pred[algo][1]()

    # Run 5-fold cross-validation and print results
    algo.fit(train_set)
    predictions = algo.test(test_set)

    for it, dirs in enumerate(dir_list):
        if not path.isdir("/".join(dir_list[:it + 1])):
            os.makedirs("/".join(dir_list[:it + 1]))

    metric_list = ['rmse', 'mae', 'mse', 'fcp']
    for met_name in metric_list:
        with open(pred[algo][0] + '/' + met_name + '.txt', 'w') as f:
            score = met[met_name](predictions)
            f.write(str(score))
            pd.DataFrame([[score]], columns=[met_name]) \
                .to_csv(pred[algo][0] + "/" + met_name + ".tsv", index_label='index', sep='\t')
