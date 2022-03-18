import os
import os.path as path
import pandas as pd
import yaml
from surprise import accuracy
from objects.BookLib import BookLib
from surprise.model_selection import train_test_split
from surprise import prediction_algorithms
from surprise.model_selection import LeaveOneOut
from objects.Metrics import Metrics
from surprise import KNNBaseline
from surprise.model_selection import KFold
from pyparsing import Word, OneOrMore, Optional, Group, Combine, Suppress, alphanums, alphas, nums, srange

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

n = 10  # TopN
book_library = BookLib()
data, rankings = book_library()
fulltrain = data.build_full_trainset()
topNPredicted = None
train_set, test_set = train_test_split(data, test_size=0.025)


pred = {'KNNBasicPearson': ['metrics/knnBasicNoShrink', prediction_algorithms.knns.KNNBasic]}
for algo_name in pred.keys():
    algo = None
    if algo_name == 'KNNBasicPearson':
        sim_options = {'name': 'pearson'}
        algo = pred[algo_name][1](sim_options=sim_options)

    algo.fit(fulltrain)
    predictions = algo.predict(uid=user_encoded2user[123], iid=book_encoded2book[324])
print(type(str(predictions)), str(predictions))
t = str(predictions)
data_usr = {}
word = Word(alphanums)
colon = Suppress(":")
word = Word(alphanums)
command = Group(OneOrMore(word))
device = Group(OneOrMore(word))
argument = Group(OneOrMore(word))
digits = Word(nums).set_name("numeric digits")
event = "user: " + Word(alphanums) + "item: " + Word(alphanums) + "r_ui = " + Word(alphanums) + 'est = ' \
        + Combine(digits + '.' + digits) + Group(OneOrMore(Word(srange("[{}]"))))
d = event.parseString(t)
print(d)
data_usr.update({d[3]: d[7]})
print(data_usr)
print(len(book_encoded2book))
print(len(user_encoded2user))
