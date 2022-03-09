import argparse
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from pandas.core.frame import DataFrame
from io import StringIO
from sklearn.model_selection import train_test_split
from objects.RecommenderNet import RecommenderNet  # add . for AzFunc
from tensorflow import keras

global metadata_path
global test_ratio
global sample
global epochs
global lr
global rand_seed
global ratio_val_test
global EMBEDDING_SIZE
global user2user_encoded
global book2book_encoded
global book_encoded2book
global userencoded2user
global min_rating
global max_rating
global num_users
global num_books
global x_train
global y_train
global x_val
global y_val
global x_test
global y_test
global index_for_val
global index_for_test
global df

metadata_path = None
test_ratio = None
sample = None
epochs = None
lr = None
rand_seed = None
ratio_val_test = None
EMBEDDING_SIZE = None
user2user_encoded = None
book2book_encoded = None
book_encoded2book = None
userencoded2user = None
min_rating = None
max_rating = None
num_users = None
num_books = None
x_train = None
y_train = None
x_val = None
y_val = None
x_test = None
y_test = None
index_for_val = None
index_for_test = None
df = None


def load_db():
    global df
    global metadata_path
    global test_ratio
    global sample
    global epochs
    global lr
    global rand_seed
    global ratio_val_test
    global EMBEDDING_SIZE
    global user2user_encoded
    global book2book_encoded
    global book_encoded2book
    global userencoded2user
    global min_rating
    global max_rating
    global num_users
    global num_books
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        sample = int(params['collab']['sample'])
        ratio_val_test = float(params['collab']['ratio_val_test'])
        epochs = int(params['collab']['epochs'])
        lr = float(params['collab']['learning_rate'])
        metadata_path = str(params['data']['metadata'])
        book_db = str(params['data']['ranking'])
        rand_seed = int(params['collab']['rand_seed'])
        EMBEDDING_SIZE = int(params['collab']['embedding_size'])
    dataframe = pd.read_csv(book_db, index_col='index')
    if not sample == 0:
        dataframe = dataframe.sample(n=sample, random_state=rand_seed)
    print('db -> ' + str(dataframe.shape[0]))
    user_ids = dataframe["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    book_ids = dataframe["click_article_id"].unique().tolist()
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    book_encoded2book = {i: x for i, x in enumerate(book_ids)}
    dataframe["user"] = dataframe["user_id"].map(user2user_encoded)
    dataframe["book"] = dataframe["click_article_id"].map(book2book_encoded)

    num_users = len(user2user_encoded)
    num_books = len(book_encoded2book)
    dataframe["rating"] = dataframe["rating"].values.astype(np.float32)
    min_rating = min(dataframe["rating"])
    max_rating = max(dataframe["rating"])

    description = pd.DataFrame([[num_users, num_books, min_rating, max_rating, sample]],
                               columns=["users", "books", "Min_rating", "max_rating", "lines"])
    print(description)
    dataframe.sort_values(by=['user_id', 'click_timestamp', 'rating'])
    dataframe.to_csv('data/books_rating.csv', index_label='index')
    df = dataframe.copy()
    return


def split_data():
    global df
    global user2user_encoded
    global book2book_encoded
    global book_encoded2book
    global user2user_encoded
    global userencoded2user
    global min_rating
    global max_rating
    global num_users
    global num_books
    global x_train
    global y_train
    global x_val
    global y_val
    global x_test
    global y_test
    global index_for_val
    global index_for_test

    if not type(df) == DataFrame:
        load_db()

    x_max = df["user"].max()  # Number of clients in base
    index_for_val = x_max - int(x_max*ratio_val_test*2)
    index_for_test = x_max - int(x_max*ratio_val_test)
    print(index_for_val, index_for_test, 'INDEXES')
    x_train = df.query('user <= @index_for_val')[["user", "book"]].values
    y_train = df.query('user <= @index_for_val')["rating"].apply(
        lambda x1: (x1 - min_rating) / (max_rating - min_rating)).values  # Normalization
    x_val = df.query('(user > @index_for_val) and (user <= @index_for_test)')[["user", "book"]].values
    y_val = df.query('(user > @index_for_val) and (user <= @index_for_test)')["rating"].apply(
        lambda x1: (x1 - min_rating) / (max_rating - min_rating)).values  # Normalization
    x_test = df.query('user > @index_for_test')[["user", "book"]].values
    y_test = df.query('user > @index_for_test')["rating"].apply(
        lambda x1: (x1 - min_rating) / (max_rating - min_rating)).values  # Normalization
    print('Data : X -> ' + str(x_train.shape[0]))
    return


def train(user=0):
    global num_users
    global num_books
    global x_train
    global y_train
    global x_val
    global y_val
    global x_test
    global y_test

    split_data()
    model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=lr)
    )

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val),
    )
    model.save('model')


def predict(user=0):
    global metadata_path
    global user2user_encoded
    global book2book_encoded
    global book_encoded2book
    global user2user_encoded
    global userencoded2user
    global min_rating
    global max_rating
    global num_users
    global num_books
    global df

    if not type(df) == DataFrame:
        load_db()

    model = keras.models.load_model('model')
    book_df = pd.read_csv(metadata_path)
    user = int(user)
    if user == 0:
        print('tirage au sort !')
        user = df.user_id.sample(1).iloc[0]
    books_watched_by_user = df.loc[df['user_id'] == user]
    print(books_watched_by_user, 'BKs Wtch')
    if books_watched_by_user.empty:
        print('pas d’historique pour cet utilisateur')
        sys.exit()
    books_not_watched = book_df[
        ~book_df["article_id"].isin(books_watched_by_user.click_article_id.values)
    ]["article_id"]
    books_not_watched = list(
        set(books_not_watched).intersection(set(book2book_encoded.keys()))
    )
    books_not_watched = [[book2book_encoded.get(x)] for x in books_not_watched]
    user_encoder = user2user_encoded.get(user)
    user_book_array = np.hstack(
        ([[user_encoder]] * len(books_not_watched), books_not_watched)
    )
    ratings = model.predict(user_book_array).flatten()
    top_ratings_indices = ratings.argsort()[-5:][::-1]
    recommended_book_ids = [
        book_encoded2book.get(books_not_watched[x][0]) for x in top_ratings_indices
    ]

    try:
        recommended_books = book_df[book_df["article_id"].isin(recommended_book_ids)]
        recomm = []
        n = 0
        for row in recommended_books.itertuples():
            n += 1            
            recomm.append(str(row.article_id))
        myJSON = [str(x) for x in recomm]

        myArray = StringIO()
        json.dump(myJSON, myArray)
        print(myArray.getvalue())
        return myArray.getvalue()

    except KeyError as e:
        print('Utilisateur sans article concordant avec la matrice tfidf')
        return ('Utilisateur sans article concordant avec la matrice tfidf')


def score(user=0):
    global metadata_path
    global user2user_encoded
    global book2book_encoded
    global book_encoded2book
    global user2user_encoded
    global userencoded2user
    global min_rating
    global max_rating
    global num_users
    global num_books
    global x_test
    global y_test
    global df
    if not type(df) == DataFrame:
        load_db()
    print('user', user)
    model = keras.models.load_model('model')
    book_df = pd.read_csv(metadata_path)

    if user == 0:
        print('tirage au sort !')
        user = df.user_id.sample(1).iloc[0]  #
    y_true = df.loc[df['user_id'] == user]['click_article_id'].tolist()[:-1]  # Affectation des autres livres en y_true
    y_true = [str(x) for x in y_true]
    books_watched_by_user = DataFrame(df.loc[df['user_id'] == user])#.iloc[-1]).T  # Sélection d’un seul livre
    print(books_watched_by_user)
    if books_watched_by_user.empty:
        print('pas d’historique pour cet utilisateur')
        sys.exit()
    books_not_watched = book_df[
        ~book_df["article_id"].isin(books_watched_by_user.click_article_id.values)
    ]["article_id"]
    books_not_watched = list(
        set(books_not_watched).intersection(set(book2book_encoded.keys()))
    )
    books_not_watched = [[book2book_encoded.get(x)] for x in books_not_watched]
    user_encoder = user2user_encoded.get(user)
    user_book_array = np.hstack(
        ([[user_encoder]] * len(books_not_watched), books_not_watched)
    )
    ratings = model.predict(user_book_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_book_ids = [
        book_encoded2book.get(books_not_watched[x][0]) for x in top_ratings_indices
    ]

    try:
        recommended_books = book_df[book_df["article_id"].isin(recommended_book_ids)]
        recomm = []
        n = 0
        for row in recommended_books.itertuples():
            n += 1
            recomm.append(str(row.article_id))
        y_pred = [str(x) for x in recomm]
       # y_true = [str(int(x)) for x in books_watched_by_user['click_article_id'].tolist()]
        score_pred = sum([x in y_true for x in y_pred])
        print(y_pred)
        print(y_true)
        print(score_pred)
        return score_pred

    except KeyError as e:
        print('Utilisateur sans article concordant avec la matrice tfidf')
        return ('Utilisateur sans article concordant avec la matrice tfidf')


def scoring(param=0):
    global x_test
    global y_test
    global index_for_test
    global df
    if not type(df) == DataFrame:
        load_db()
    if index_for_test == None:
        split_data()
    score_array = []
    for usnb in df.query('user > @index_for_test')['user_id'].unique()[-2:]:  # enlever le slicer après le mode dev
        score_array.append(score(usnb))
    df_score = pd.DataFrame(score_array, columns=['score'])
    df_score.to_csv('score.csv', index_label='index')

if __name__ == "__main__":
    function_map = {
        'predict': predict,
        'train': train,
        'score': score,
        'scoring': scoring
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('param')
    args = parser.parse_args()
    function = function_map[args.command]
    param = int(args.param)
    function(param)

