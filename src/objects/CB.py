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
        matrix_size = int(params['machine_learning']['matrix_size'])
        clicks = str(params['data']['clicks'])
        clicks_agg = str(params['data']['clicks_agg'])
        book_db = str(params['data']['ranking'])
    clicks_list = glob(clicks + '/*.csv')
    clicks = choice(clicks_list)  # Sélection aléatoire d’un fichier de clicks
    return embeddings_path, metadata_path, matrix_size, clicks, clicks_list, clicks_agg, book_db


def build_user_profile(person_id):
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
    usr_ds = pd.read_csv(clicks_agg)
    ds = pd.read_csv(metadata_path)
    tfidf_matrix = pd.read_pickle(embeddings_path)
    interactions_person_df = usr_ds.loc[(usr_ds['user_id'] == int(person_id))]
    interactions_person_df = interactions_person_df.loc[interactions_person_df['click_article_id'].isin(ds['article_id'])]
    # Le profil contiendra les vecteurs de poids de la matrice TF-IDF
    user_profiles = {}
    # Nous listons les articles déjà consultés
    ids = interactions_person_df['click_article_id'].values[0]
    # Nous nous assurons que le résultat est sous forme de liste
    ids = ids if (type(ids) == list) else [ids]
    # Nous attrapons les poids relatifs à nos items dans la matrice TF-IDF
    item_profiles_list = [tfidf_matrix[x:x + 1] for x in ids]
    # Nous les disposons en colonnes
    item_profiles = np.stack(item_profiles_list)
    user_item_profiles = item_profiles
    user_item_strengths = [max(np.array(interactions_person_df['session_size']))]
    user_item_strengths_weighted_avg = np.sum(np.multiply(user_item_profiles,
                                                          user_item_strengths), axis=0) / np.sum(
        user_item_strengths)
    user_profile_norm = normalize(user_item_strengths_weighted_avg)
    user_profiles[person_id] = user_profile_norm
    return user_profiles


def similar_items_to_user_profile(person_id, topn=1000):
    global ds
    global clicks
    user_profiles = build_user_profile(person_id)
    # Computes the cosine similarity between the user profile and all item profiles
    cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
    # Gets the top similar items
    similar_indices = cosine_similarities.argsort().flatten()[-topn:]
    # Sort the similar items by similarity
    similar_items = sorted([(ds.loc[ds['article_id'] == i]['article_id'].values[0], cosine_similarities[0, i]) for i in similar_indices],
                           key=lambda x: -x[1])
    return similar_items


# def lambda_handler(event, context):
#     name = event['queryStringParameters']['name']
#     return {
#         "statusCode": 200,
#         "headers": {"Content-Type": "application/json"},
#         "body": json.dumps({"Hello": name})
#     }


def lambda_fct(user=16545):
    global ds  # Valeur Initialisée dans cos_sim
    global tfidf_matrix
    global usr_ds  # DataFrame de nos clicks
    global clicks  # Path du clicks
    global clicks_agg  # Path du fichier concaténé
    global embeddings_path
    global metadata_path
    global clicks_list
    embeddings_path, metadata_path, matrix_size, clicks, clicks_list, clicks_agg, book_db = load_params()
    try:
        recomm = similar_items_to_user_profile(user, topn=10)  # Retour des 5 meilleures recommandations
        myJSON = [str(x[0]) for x in recomm]
        # with open('src/inference_content_based_filter.txt', 'w') as f:
        #     f.write('user : ' + str(user) + '\n' + 'recommendation : ' + str([str(x[0]) for x in recomm])
        #             + '\nscore           : ' + str([str(round(x[1], 2)) for x in recomm]))
        myArray = StringIO()
        json.dump(myJSON, myArray)
        print(myJSON)
        return myArray.getvalue()
    except KeyError as e:
        return 'Utilisateur sans article concordant avec la matrice tfidf'


def main():
    lambda_fct()


if __name__ == "__main__":
    main()