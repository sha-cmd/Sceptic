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
from .RecommenderNet import RecommenderNet  # add . for AzFunc
from sklearn.model_selection import LeaveOneOut
from tensorflow import keras

class CF:
    def __init__(self):    
        self.metadata_path = None
        self.test_ratio = None
        self.sample = None
        self.epochs = None
        self.lr = None
        self.rand_seed = None
        self.ratio_val_test = None
        self.EMBEDDING_SIZE = None
        self.user2user_encoded = None
        self.book2book_encoded = None
        self.book_encoded2book = None
        self.userencoded2user = None
        self.min_rating = None
        self.max_rating = None
        self.num_users = None
        self.num_books = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.index_for_val = None
        self.index_for_test = None
        self.dataframe = None
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
            self.sample = int(params['machine_learning']['sample'])
            self.ratio_val_test = float(params['machine_learning']['ratio_val_test'])
            self.epochs = int(params['machine_learning']['epochs'])
            self.lr = float(params['machine_learning']['learning_rate'])
            self.metadata_path = str(params['data']['metadata'])
            self.book_db = str(params['data']['ranking'])
            self.rand_seed = int(params['machine_learning']['rand_seed'])
            self.EMBEDDING_SIZE = int(params['machine_learning']['embedding_size'])


    def load_db(self):
        
        self.dataframe = pd.read_csv(self.book_db, index_col='index', sep=';')
        #if not self.sample == 0:
        #self.dataframe = self.dataframe.sample(n=self.sample, random_state=self.rand_seed)
        print('db -> ' + str(self.dataframe.shape[0]))
        self.user_ids = self.dataframe["userID"].unique().tolist()
        self.user2user_encoded = {x: i for i, x in enumerate(self.user_ids)}
        self.userencoded2user = {i: x for i, x in enumerate(self.user_ids)}
        self.book_ids = self.dataframe["itemID"].unique().tolist()
        self.book2book_encoded = {x: i for i, x in enumerate(self.book_ids)}
        self.book_encoded2book = {i: x for i, x in enumerate(self.book_ids)}
        self.dataframe["user"] = self.dataframe["userID"].map(self.user2user_encoded)
        self.dataframe["book"] = self.dataframe["itemID"].map(self.book2book_encoded)

        self.num_users = len(self.user2user_encoded)
        self.num_books = len(self.book_encoded2book)
        self.dataframe["rating"] = self.dataframe["rating"].values.astype(np.float32)
        self.min_rating = min(self.dataframe["rating"])
        self.max_rating = max(self.dataframe["rating"])

        description = pd.DataFrame([[self.num_users, self.num_books, self.min_rating, self.max_rating, self.sample]],
                                   columns=["users", "books", "Min_rating", "max_rating", "lines"])
        print(description)
        #dataframe.sort_values(by=['user_id', 'click_timestamp', 'rating'])
        #dataframe.to_csv('data/books_rating.csv', index_label='index')
        #self.dataframe = dataframe.copy()
        return


    def split_data(self):
        
        if not type(self.dataframe) == DataFrame:
            self.load_db()

        x_max = self.dataframe["user"].max()  # Number of clients in base
        index_for_val = x_max - int(x_max*self.ratio_val_test*2)
        index_for_test = x_max - int(x_max*self.ratio_val_test)
        print(index_for_val, index_for_test, 'INDEXES')
        self.x_train = self.dataframe.query('user <= @index_for_val')[["user", "book"]].values
        self.y_train = self.dataframe.query('user <= @index_for_val')["rating"].apply(
            lambda x1: (x1 - self.min_rating) / (self.max_rating - self.min_rating)).values  # Normalization
        self.x_val = self.dataframe.query('(user > @index_for_val) and (user <= @index_for_test)')[["user", "book"]].values
        self.y_val = self.dataframe.query('(user > @index_for_val) and (user <= @index_for_test)')["rating"].apply(
            lambda x1: (x1 - self.min_rating) / (self.max_rating - self.min_rating)).values  # Normalization
        self.x_test = self.dataframe.query('user > @index_for_test')[["user", "book"]].values
        self.y_test = self.dataframe.query('user > @index_for_test')["rating"].apply(
            lambda x1: (x1 - self.min_rating) / (self.max_rating - self.min_rating)).values  # Normalization
        print('Data : X -> ' + str(self.x_train.shape[0]))
        return


    def train(self, user=0):
        self.split_data()
        model = RecommenderNet(self.num_users, self.num_books, self.EMBEDDING_SIZE)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=self.lr)
        )

        model.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=64,
            epochs=self.epochs,
            verbose=1,
            validation_data=(self.x_val, self.y_val),
        )
        model.save('model')

    def getTopN(self, user=0):

        data = {}
        if not type(self.dataframe) == DataFrame:
            self.load_db()
        #user = self.userencoded2user[user]
        model = keras.models.load_model('model')
        book_df = pd.read_csv(self.metadata_path)
        nb_user = len(self.userencoded2user)
        print(nb_user)
        for user in range(5):
            user = self.userencoded2user[user]
            user = int(user)
            #if user == 0:  # Dev Mode
            #    print('tirage au sort !')
            #    user = self.dataframe.userID.sample(1).iloc[0]
            #    print(user)
            books_watched_by_user = self.dataframe.loc[self.dataframe['userID'] == user]
            if books_watched_by_user.empty:
                print('pas d’historique pour cet utilisateur')
                sys.exit()
            books_not_watched = book_df[
                ~book_df["article_id"].isin(books_watched_by_user.itemID.values)
            ]["article_id"]
            books_not_watched = list(
                set(books_not_watched).intersection(set(self.book2book_encoded.keys()))
            )
            books_not_watched = [[self.book2book_encoded.get(x)] for x in books_not_watched]
            user_encoder = self.user2user_encoded.get(user)
            user_book_array = np.hstack(
                ([[user_encoder]] * len(books_not_watched), books_not_watched)
            )
            ratings = model.predict(user_book_array).flatten()
            top_ratings_indices = ratings.argsort()[-10:][::-1]
            recommended_book_ids = [
                self.book_encoded2book.get(books_not_watched[x][0]) for x in top_ratings_indices
            ]
            recommended_book_est = [
                ratings[x]*5 for x in top_ratings_indices
            ]
            try:
                recommended_books = book_df[book_df["article_id"].isin(recommended_book_ids)]
                recomm = []
                n = 0
                for row in recommended_books.itertuples():
                    recomm.append((str(row.article_id), str(recommended_book_est[n])))
                    n += 1
                data.update({user: recomm})
                myJSON = [str(x) for x in recomm]
                myArray = StringIO()
                json.dump(myJSON, myArray)
                #print(myArray.getvalue())
            #return myArray.getvalue()

            except KeyError as e:
                print('Utilisateur sans article concordant avec la matrice tfiself.dataframe')
                return ('Utilisateur sans article concordant avec la matrice tfiself.dataframe')

        print(data)

    def predict(self, user=0):
        if not type(self.dataframe) == DataFrame:
            self.load_db()
        user = self.userencoded2user[user]
        model = keras.models.load_model('model')
        book_df = pd.read_csv(self.metadata_path)
        user = int(user)
        if user == 0:
            print('tirage au sort !')
            user = self.dataframe.userID.sample(1).iloc[0]
            print(user)
        books_watched_by_user = self.dataframe.loc[self.dataframe['userID'] == user]
        if books_watched_by_user.empty:
            print('pas d’historique pour cet utilisateur')
            sys.exit()
        books_not_watched = book_df[
            ~book_df["article_id"].isin(books_watched_by_user.itemID.values)
        ]["article_id"]
        books_not_watched = list(
            set(books_not_watched).intersection(set(self.book2book_encoded.keys()))
        )
        books_not_watched = [[self.book2book_encoded.get(x)] for x in books_not_watched]
        user_encoder = self.user2user_encoded.get(user)
        user_book_array = np.hstack(
            ([[user_encoder]] * len(books_not_watched), books_not_watched)
        )
        ratings = model.predict(user_book_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]  # Retour du topn des meilleures recommandations
        recommended_book_ids = [
            self.book_encoded2book.get(books_not_watched[x][0]) for x in top_ratings_indices
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
            print('Utilisateur sans article concordant avec la matrice tfiself.dataframe')
            return ('Utilisateur sans article concordant avec la matrice tfiself.dataframe')

    def score(self, user=0):
        if not type(self.dataframe) == DataFrame:
            self.load_db()
        print('user', user)
        model = keras.models.load_model('model')
        book_df = pd.read_csv(self.metadata_path)

        if user == 0:
            print('tirage au sort !')
            user = self.dataframe.user_id.sample(1).iloc[0]  #
        y_true = self.dataframe.loc[self.dataframe['user_id'] == user]['click_article_id'].tolist()[:-1]  # Affectation des autres livres en y_true
        y_true = [str(x) for x in y_true]
        books_watched_by_user = DataFrame(self.dataframe.loc[self.dataframe['user_id'] == user])#.iloc[-1]).T  # Sélection d’un seul livre
        print(books_watched_by_user)
        if books_watched_by_user.empty:
            print('pas d’historique pour cet utilisateur')
            sys.exit()
        books_not_watched = book_df[
            ~book_df["itemID"].isin(books_watched_by_user.click_article_id.values)
        ]["itemID"]
        books_not_watched = list(
            set(books_not_watched).intersection(set(self.book2book_encoded.keys()))
        )
        books_not_watched = [[self.book2book_encoded.get(x)] for x in books_not_watched]
        user_encoder = self.user2user_encoded.get(user)
        user_book_array = np.hstack(
            ([[user_encoder]] * len(books_not_watched), books_not_watched)
        )
        ratings = model.predict(user_book_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_book_ids = [
            self.book_encoded2book.get(books_not_watched[x][0]) for x in top_ratings_indices
        ]

        try:
            recommended_books = book_df[book_df["itemID"].isin(recommended_book_ids)]
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
            print('Utilisateur sans article concordant avec la matrice tfiself.dataframe')
            return ('Utilisateur sans article concordant avec la matrice tfiself.dataframe')

    def scoring(self, param=0):
        if not type(self.dataframe) == DataFrame:
            self.load_db()
        if self.index_for_test == None:
            self.split_data()
        score_array = []
        for usnb in self.dataframe.query('user > @index_for_test')['userID'].unique()[-2:]:  # enlever le slicer après le mode dev
            score_array.append(self.score(usnb))
        self.dataframe_score = pd.DataFrame(score_array, columns=['score'])
        self.dataframe_score.to_csv('score.csv', index_label='index')

if __name__ == "__main__":
    cf = CF()
    function_map = {
        'predict': cf.predict,
        'train': cf.train,
        'score': cf.score,
        'scoring': cf.scoring,
        'getTopN': cf.getTopN
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('param')
    args = parser.parse_args()
    function = function_map[args.command]
    param = int(args.param)
    function(param)
    