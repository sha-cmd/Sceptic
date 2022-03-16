import os
import os.path as path
from surprise import SVD
from surprise import accuracy
from surprise import Dataset
from surprise.model_selection import cross_validate
from objects.BookLib import BookLib
from surprise.model_selection import train_test_split
from surprise import prediction_algorithms

booklibrary = BookLib()
data = booklibrary()
trainset, testset = train_test_split(data, test_size=.25)


# Then compute RMSE
met = {'rmse': accuracy.rmse,
       'mae': accuracy.mae,
       'mse': accuracy.mse,
       'fcp': accuracy.fcp}
pred = {'NormalPred': ['metrics/normalPred', prediction_algorithms.random_pred.NormalPredictor],
        'baseLineALS': ['metrics/baseLineALS', prediction_algorithms.baseline_only.BaselineOnly],
        'baseLineSGD': ['metrics/baseLineSGD', prediction_algorithms.baseline_only.BaselineOnly],
        'KNNBasicALS': ['metrics/knnBasicALS', prediction_algorithms.knns.KNNBasic],
        'KNNBasicSims': ['metrics/knnBasicSims', prediction_algorithms.knns.KNNBasic],
        'KNNBasicNoShrink': ['metrics/knnBasicNoShrink', prediction_algorithms.knns.KNNBasic],
        'SVD': ['metrics/SVD', prediction_algorithms.matrix_factorization.SVD],
        'SVDpp': ['metrics/SVDpp', prediction_algorithms.matrix_factorization.SVDpp],
        'NMF': ['metrics/NMF', prediction_algorithms.matrix_factorization.NMF]
}



for a in pred.keys():
    b = pred[a][0].split('/')
    algo = None
    print(a)
    # We'll use the famous SVD algorithm.
    if a == 'baseLineALS':
        bsl_options = {'method': 'als',
                       'n_epochs': 5,
                       'reg_u': 12,
                       'reg_i': 5
                       }
        algo = pred[a][1](bsl_options=bsl_options)
    elif a == 'baseLineSGD':
        bsl_options = {'method': 'sgd',
                       'learning_rate': .00005,
                       }
        algo = pred[a][1](bsl_options=bsl_options)
    elif a == 'KNNBasicALS':
        bsl_options = {'method': 'als',
                       'n_epochs': 20,
                       }
        sim_options = {'name': 'pearson_baseline'}
        algo = pred[a][1](bsl_options=bsl_options, sim_options=sim_options)
    elif a == 'KNNBasicSims':
        sim_options = {'name': 'cosine',
                       'user_based': False  # compute  similarities between items
                       }
        algo = pred[a][1](sim_options=sim_options)
    elif a == 'KNNBasicNoShrink':
        sim_options = {'name': 'pearson_baseline',
                       'shrinkage': 0  # no shrinkage
                       }
        algo = pred[a][1](sim_options=sim_options)
    else:
        algo = pred[a][1]()

    # Run 5-fold cross-validation and print results
    algo.fit(trainset)
    predictions = algo.test(testset)

    for it, dirs in enumerate(b):
        if not path.isdir("/".join(b[:it + 1])):
            os.makedirs("/".join(b[:it + 1]))

    c = ['rmse', 'mae', 'mse', 'fcp']
    for met_name in c:
        with open(pred[a][0] + '/' + met_name + '.txt', 'w') as f:
            f.write(str(met[met_name](predictions)))
