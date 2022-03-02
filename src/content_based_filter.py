import modin.pandas as md
import numpy as np
import os
import pandas as pd
import yaml

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from random import choice
from glob import glob

global ds  # Valeur Initialisée dans cos_sim
global tfidf_matrix
global usr_ds  # DataFrame de nos clicks
global clicks  # Path du clicks
global clicks_agg  # Path du fichier concaténé


def load_params():
    global clicks
    global clicks_agg
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        embeddings_path = str(params['data']['embeddings'])
        metadata_path = str(params['data']['metadata'])
        matrix_size = int(params['content_filter']['matrix_size'])
        clicks = str(params['data']['clicks'])
        clicks_agg = str(params['data']['clicks_agg'])
    clicks_list = glob(clicks + '/*.csv')
    clicks = choice(clicks_list)  # Sélection aléatoire d’un fichier de clicks
    global ds
    global tfidf_matrix
    global usr_ds
    usr_ds = md.read_csv(clicks_list[0])[['user_id', 'click_article_id', 'session_size']]
    if not os.path.isfile(clicks_agg):
        for i in range(1, len(clicks_list)):
            usr_ds = md.concat([usr_ds[['user_id', 'click_article_id', 'session_size']], md.read_csv(clicks_list[i])[['user_id', 'click_article_id', 'session_size']]])
        usr_ds.to_csv(clicks_agg)
    else:
        pass
    ds = md.read_csv(metadata_path).iloc[:matrix_size, :]
    tfidf_matrix = pd.read_pickle(embeddings_path)[:matrix_size, :]
    return embeddings_path, metadata_path, matrix_size, clicks


def build_users_profiles():
    global usr_ds
    interactions_indexed_df = usr_ds[usr_ds['click_article_id'].isin(ds['article_id'])]
    user_profiles = {}
    for person_id in interactions_indexed_df.user_id.unique():
        interactions_person_df = interactions_indexed_df \
            .loc[interactions_indexed_df['user_id'] \
                 == person_id]
        ids = interactions_person_df['click_article_id'].values[0]
        ids = ids if (type(ids) == list) else [ids]
        item_profiles_list = [tfidf_matrix[x:x + 1] for x in ids]
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
    user_profiles = build_users_profiles()
    # Computes the cosine similarity between the user profile and all item profiles
    cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
    # Gets the top similar items
    similar_indices = cosine_similarities.argsort().flatten()[-topn:]
    # Sort the similar items by similarity
    similar_items = sorted([(ds.loc[ds['article_id'] == i]['article_id'].values[0], cosine_similarities[0, i]) for i in similar_indices],
                           key=lambda x: -x[1])
    return similar_items


def main():
    global clicks
    global clicks_agg
    global ds
    global usr_ds
    embeddings_path, metadata_path, matrix_size, clicks = load_params()
    user = 236  # Test
    usr_ds = md.read_csv(clicks_agg)
    usr_ds = usr_ds.loc[usr_ds['user_id'] == user]
    print(user)  # Affichage de l’utilisateur sélectionné
    try:
        recomm = similar_items_to_user_profile(user, topn=5)  # Retour des 5 meilleures recommandations
        print(recomm)
        with open('src/inference_content_based_filter.txt', 'w') as f:
            f.write('user : ' + str(user) + '\n' + 'recommendation : ' + str(recomm))
    except KeyError as e:
        print('Utilisateur sans article concordant avec la matrice tfidf')


if __name__ == "__main__":
    main()
