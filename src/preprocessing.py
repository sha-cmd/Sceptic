import json
import numpy as np
import os
import pandas as pd
import yaml

from io import StringIO
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from random import choice
from glob import glob

global ds  # Valeur Initialisée dans cos_sim
global tfidf_matrix
global usr_ds  # DataFrame de nos clicks
global usr_ds_ranking  # DataFrame de nos scores
global clicks  # Path du clicks
global clicks_agg  # Path du fichier concaténé
global clicks_list
global metadata_path
global embeddings_path
global client_by_book
global book_by_client


def load_params():
    global ds
    global tfidf_matrix
    global usr_ds
    global usr_ds_ranking
    global clicks
    global clicks_agg
    global clicks_list
    global metadata_path
    global embeddings_path
    global client_by_book
    global book_by_client

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        embeddings_path = str(params['data']['embeddings'])
        metadata_path = str(params['data']['metadata'])
        clicks = str(params['data']['clicks'])
        clicks_agg = str(params['data']['clicks_agg'])
        book_db = str(params['data']['ranking'])
        book_rts = str(params['data']['ratings'])
        client_by_book = int(params['machine_learning']['client_by_book'])
        book_by_client = int(params['machine_learning']['book_by_client'])
    clicks_list = glob(clicks + '/*.csv')
    clicks = choice(clicks_list)  # Sélection aléatoire d’un fichier de clicks
    return embeddings_path, metadata_path, clicks, clicks_list, clicks_agg, book_db, book_rts


def build_db():
    global metadata_path
    global embeddings_path
    global clicks_agg
    global usr_ds
    global usr_ds_ranking
    global ds
    global tfidf_matrix
    global clicks
    global clicks_list
    global client_by_book
    global book_by_client

    embeddings_path, metadata_path, clicks, clicks_list, clicks_agg, book_db, book_rts = load_params()
    if not os.path.isfile(clicks_agg):
        usr_ds = pd.read_csv(clicks_list[0])[['user_id', 'click_article_id', 'session_size', 'click_timestamp']]
        for i in range(1, len(clicks_list)):
            usr_ds = pd.concat([usr_ds[['user_id', 'click_article_id', 'session_size', 'click_timestamp']],
                                pd.read_csv(clicks_list[i])[
                                    ['user_id', 'click_article_id', 'session_size', 'click_timestamp']]])
            usr_ds.index.name = 'index'
        list_usr = usr_ds['user_id'].value_counts().reset_index().rename(columns=
                                                                         {'user_id': 'book_count_usr',
                                                                          'index': 'user_id'})
        list_usr.index.name = 'index'
        list_usr = list_usr.loc[list_usr['book_count_usr'] > book_by_client]['user_id'].to_list()

        list_book = usr_ds['click_article_id'].value_counts().reset_index().rename(columns=
                                                                                   {
                                                                                       'click_article_id': 'click_count_book',
                                                                                       'index': 'click_article_id'})
        list_book.index.name = 'index'
        list_book = list_book.loc[list_book['click_count_book'] > client_by_book]['click_article_id'].to_list()
        usr_ds = usr_ds.loc[usr_ds['user_id'].isin(list_usr)]  # Elimine les clients sans plus de 5 livres
        usr_ds = usr_ds.loc[usr_ds['click_article_id'].isin(list_book)]  # Elimine les livres sans plus d’un client
        usr_ds.to_csv(clicks_agg)
    else:
        usr_ds = pd.read_csv(clicks_agg)
    if not os.path.isfile(book_db):
        usr_ds['rating'] = 1

        usr_ds_ranking = usr_ds.groupby(['user_id', 'click_article_id']).agg(
            {'click_timestamp': ['max'], 'rating': ['sum']})
        usr_ds_ranking = usr_ds_ranking.droplevel(level=1, axis=1).reset_index()
        usr_ds_ranking['rating'] = usr_ds_ranking['rating'].astype('float')
        usr_ds_ranking.index.name = 'index'
        usr_ds_ranking['rating'] = usr_ds_ranking['rating'].where(usr_ds_ranking['rating'] < 5, 5)
        usr_ds_ranking.to_csv(book_db, index_label='index')
        # Encodage des noms et des index des clients et des items
    if not os.path.isfile(book_rts):
        dataframe = pd.read_csv(book_db, index_col='index')
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
        sample = dataframe.shape[0]

        description = pd.DataFrame([[num_users, num_books, min_rating, max_rating, sample]],
                                   columns=["users", "books", "Min_rating", "max_rating", "lines"])
        print(description)
        dataframe.sort_values(by=['user_id', 'click_timestamp', 'rating'])
        dataframe.to_csv('data/books_ratings.csv', index_label='index')
        dataframe = dataframe[['user_id', 'click_article_id', 'rating', 'click_timestamp']]  # Dev mode
        dataframe = dataframe.rename(columns={'user_id': 'userID', 'click_article_id': 'itemID', 'click_timestamp':'timestamp'})
        # Elimination des clients n’ayant pas de rating différent de 1
        #files = dataframe.groupby('userID')['rating'].agg(['sum', 'size'])
        #dataframe = dataframe.loc[~dataframe['userID'].isin(files.where(files['sum'] == files['size']).dropna().index.tolist())]
        #files = dataframe.groupby('itemID')['rating'].agg(['sum', 'size'])
        #dataframe = dataframe.loc[~dataframe['itemID'].isin(files.where(files['sum'] == files['size']).dropna().index.tolist())]
        #nb = (dataframe['rating'].value_counts().iloc[0] - dataframe['rating'].value_counts().iloc[1:].max()/2)
        #black_list = (dataframe.loc[dataframe['rating'] == 1].sample(nb).index.to_list())
        #dataframe = dataframe.drop(index=black_list, axis=0)
        dataframe.to_csv('data/database.csv', index_label='index', sep=';')

def main():
    global ds  # Valeur Initialisée dans cos_sim
    global tfidf_matrix
    global usr_ds  # DataFrame de nos clicks
    global clicks  # Path du clicks
    global clicks_agg  # Path du fichier concaténé
    global embeddings_path
    global metadata_path
    global clicks_list
    embeddings_path, metadata_path, clicks, clicks_list, clicks_agg, book_db, book_rts = load_params()
    build_db()


if __name__ == "__main__":
    main()
