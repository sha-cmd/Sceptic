"""Travaux d’exploration de nos algorithmes via la bibliothèque Surprise, avec l’ajout de métrique dédié à la
recherche de performance. L’utilisation de LeaveOneOut est optimisé par la bibliothèque précédente."""

import os
import os.path as path
import pandas as pd
from surprise import accuracy
from objects.BookLib import BookLib
from surprise.model_selection import train_test_split
from surprise import prediction_algorithms
from surprise.model_selection import LeaveOneOut
from objects.Metrics import Metrics
from surprise import KNNBaseline

n = 10  # TopN
book_library = BookLib()
data, rankings = book_library()
fulltrain = data.build_full_trainset()
train_set, test_set = train_test_split(data, test_size=.25)
LOOCV = LeaveOneOut(n_splits=1, random_state=42, min_n_ratings=3)
for train, test in LOOCV.split(data):
    LOOCVTrain = train
    LOOCVTest = test
LOOCVAntiTestSet = LOOCVTrain.build_anti_testset()
# Compute similarty matrix between items so we can measure diversity
sim_options = {'name': 'cosine', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fulltrain)

met = {'rmse': accuracy.rmse,
       'mae': accuracy.mae,
       'mse': accuracy.mse,
       'fcp': accuracy.fcp,
       'TopN': Metrics.GetTopN,
       'HR': Metrics.HitRate,
       'cHR': Metrics.CumulativeHitRate,
       'ARHR': Metrics.AverageReciprocalHitRank,
       'Coverage': Metrics.UserCoverage,
       'Diversity': Metrics.Diversity,
       'Novelty': Metrics.Novelty}



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

# Utilisation de DVC requise pour un usage de ces données en format tableau. Affichage des performances en graphiques.
data = {'name': [], 'rmse': [], 'mae': [], 'mse': [], 'fcp': [], 'HR': [], 'cHR': [],
        'ARHR': [], 'Coverage': [], 'Diversity': [], 'Novelty': []}

for algo_name in pred.keys():
    dir_list = pred[algo_name][0].split('/')
    print(algo_name)
    algo = None
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

    algo.fit(train_set)
    predictions = algo.test(test_set)
    leftOutPredictions = algo.test(LOOCVTest)
    allPredictions = algo.test(LOOCVAntiTestSet)
    topNPredicted = Metrics.GetTopN(allPredictions, n, 2.0)
    for it, dirs in enumerate(dir_list):
        if not path.isdir("/".join(dir_list[:it + 1])):
            os.makedirs("/".join(dir_list[:it + 1]))

    metric_list = ['rmse', 'mae', 'mse', 'fcp', 'HR', 'cHR', 'ARHR', 'Coverage', 'Diversity', 'Novelty']
    json = {}
    for met_name in metric_list:
        print(met_name)
        with open(pred[algo_name][0] + '/' + met_name + '.txt', 'w') as f:
            score = None
            if met_name in ['rmse', 'mae', 'mse', 'fcp']:
                score = met[met_name](predictions)
            elif met_name == 'Coverage':
                score = met[met_name](topNPredicted, fulltrain.n_users, ratingThreshold=2.0)
            elif met_name == 'Diversity':
                score = met[met_name](topNPredicted, simsAlgo)
            elif met_name == 'Novelty':
                score = met[met_name](topNPredicted, rankings)
            else:
                score = met[met_name](topNPredicted, leftOutPredictions)
            f.write(str(score))
            pd.DataFrame([[score]], columns=[met_name]) \
                .to_csv(pred[algo_name][0] + "/" + met_name + ".tsv", index_label='index', sep='\t')
        json[met_name].append(score)
        with open(pred[algo_name][0] + '/' + met_name + '.json', 'w') as f:
            chain = ''
            nb = len(data.keys())
            n = 0
            for key, value in data.items():
                chain += '"' + key + '": ' + value
                if n < nb:
                    chain += ','
                n += 1
            f.writelines('{' + chain + '}')
        data[met_name].append(score)
    data['name'].append(algo_name)

# Maîtrise des résultats avec projection en numéraire dans un fichier à partir d’un dictionnaire
pd.DataFrame.from_dict(data).to_csv('metrics/synthese.csv', index_label='index', sep=';')
