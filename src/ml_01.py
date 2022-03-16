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
pred = {'metrics/normalPred': prediction_algorithms.random_pred.NormalPredictor,
        'metrics/baseLine': prediction_algorithms.baseline_only.BaselineOnly}

for a in pred.keys():
    b = a.split('/')

    # We'll use the famous SVD algorithm.
    algo = pred[a]()

    # Run 5-fold cross-validation and print results
    algo.fit(trainset)
    predictions = algo.test(testset)

    for it, dirs in enumerate(b):
        if not path.isdir("/".join(b[:it + 1])):
            os.makedirs("/".join(b[:it + 1]))

    c = ['rmse', 'mae', 'mse', 'fcp']
    for met_name in c:
        with open(a + '/' + met_name + '.txt', 'w') as f:
            f.write(str(met[met_name](predictions)))
