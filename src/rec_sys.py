
import pandas as pd
import yaml
from joblib import Parallel, delayed
from objects.BookLib import BookLib
from surprise import prediction_algorithms

from pyparsing import Word, OneOrMore, Group, Combine, alphanums, nums, srange

global data_rs
data_rs = {}


def data_calc(raw_uid, raw_iid, algorithme):
    global data_rs
    digits = Word(nums)
    event = "user: " + Word(alphanums) + "item: " + Word(alphanums) + "r_ui = " + Word(alphanums) + 'est = ' \
            + Combine(digits + '.' + digits) + Group(OneOrMore(Word(srange("[{}]"))))
    d = event.parseString(str(algorithme.predict(uid=raw_uid, iid=raw_iid)))
    if not data.get(raw_uid) == None:
        data_rs[raw_uid].update({d[3]: d[7]})
    else:
        data_rs.update({raw_uid: {d[3]: d[7]}})

        
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

    book_db = str(params['data']['ranking'])
dataframe = pd.read_csv(book_db, index_col='index', sep=';')
print('db -> ' + str(dataframe.shape[0]))
user_ids = dataframe["userID"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
book_ids = dataframe["itemID"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded2book = {i: x for i, x in enumerate(book_ids)}
dataframe["user"] = dataframe["userID"].map(user2user_encoded)
dataframe["book"] = dataframe["itemID"].map(book2book_encoded)

book_library = BookLib()
data, rankings = book_library()
fulltrain = data.build_full_trainset()

pred = {'KNNBasicPearson': ['metrics/knnBasicNoShrink', prediction_algorithms.knns.KNNBasic]}
for algo_name in pred.keys():
    algo = None
    if algo_name == 'KNNBasicPearson':
        sim_options = {'name': 'pearson'}
        algo = pred[algo_name][1](sim_options=sim_options)

    algo.fit(fulltrain)

    # Calcul de la matrice des recommendations pour chaque couple user/item
    for ruid in dataframe['userID'].unique():
        Parallel(n_jobs=-1)(
                delayed(data_calc)(ruid, riid, algo) for riid in dataframe['itemID'].unique())

df = pd.DataFrame.from_dict(data_rs)