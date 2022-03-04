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

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        embeddings_path = str(params['data']['embeddings'])
        metadata_path = str(params['data']['metadata'])
        matrix_size = int(params['content_filter']['matrix_size'])
        clicks = str(params['data']['clicks'])
        clicks_agg = str(params['data']['clicks_agg'])
        book_db = str(params['data']['ranking'])
    clicks_list = glob(clicks + '/*.csv')
    clicks = choice(clicks_list)  # Sélection aléatoire d’un fichier de clicks
    return embeddings_path, metadata_path, matrix_size, clicks, clicks_list, clicks_agg, book_db


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
    embeddings_path, metadata_path, matrix_size, clicks, clicks_list, clicks_agg, book_db = load_params()
    if not os.path.isfile(clicks_agg):
        usr_ds = pd.read_csv(clicks_list[0])[['user_id', 'click_article_id', 'session_size', 'session_id']]
        for i in range(1, len(clicks_list)):
            usr_ds = pd.concat([usr_ds[['user_id', 'click_article_id', 'session_size', 'session_id']],
                                pd.read_csv(clicks_list[i])[['user_id', 'click_article_id', 'session_size',
                                                             'session_id']]])
            usr_ds.index.name = 'index'
        usr_ds.to_csv(clicks_agg)
    else:
        usr_ds = pd.read_csv(clicks_agg)
    if not os.path.isfile(book_db):
        usr_ds['rating'] = 1
        usr_ds_ranking = usr_ds.groupby(['user_id', 'click_article_id']).sum()
        usr_ds_ranking = usr_ds_ranking.reset_index()
        usr_ds_ranking = usr_ds_ranking[['user_id', 'click_article_id', 'rating']]
        usr_ds_ranking['rating'] = usr_ds_ranking['rating'].astype('float')
        usr_ds_ranking.index.name = 'index'
        usr_ds_ranking.to_csv(book_db)
    else:
        usr_ds_ranking = pd.read_csv(book_db)


def main():
    global ds  # Valeur Initialisée dans cos_sim
    global tfidf_matrix
    global usr_ds  # DataFrame de nos clicks
    global clicks  # Path du clicks
    global clicks_agg  # Path du fichier concaténé
    global embeddings_path
    global metadata_path
    global clicks_list
    embeddings_path, metadata_path, matrix_size, clicks, clicks_list, clicks_agg, book_db = load_params()
    build_db()


if __name__ == "__main__":
    main()
