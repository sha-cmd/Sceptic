import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from random import choice
from glob import glob

global ds  # Valeur Initialisée dans cos_sim
global tfidf_matrix
global usr_ds  # DataFrame de nos clicks
global clicks  # Path du clicks


def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        embeddings_path = str(params['data']['embeddings'])
        metadata_path = str(params['data']['metadata'])
        matrix_size = int(params['content_filter']['matrix_size'])
        clicks = str(params['data']['clicks'])
    clicks_list = glob(clicks + '/*.csv')
    clicks = choice(clicks_list)  # Sélection aléatoire d’un fichier de clicks
    print(clicks)
    print(embeddings_path)
    print('chargement du fichier embeddings')
    global ds
    global tfidf_matrix
    global usr_ds
    usr_ds = pd.read_csv(clicks)
    ds = pd.read_csv(metadata_path).iloc[:matrix_size, :]
    tfidf_matrix = pd.read_pickle(embeddings_path)[:matrix_size, :]
    return embeddings_path, metadata_path, matrix_size, clicks


def build_users_profiles(clicks):
    global usr_ds
    interactions_indexed_df = usr_ds[usr_ds['click_article_id'].isin(ds['article_id'])]
    user_profiles = {}
    for person_id in interactions_indexed_df.user_id.unique():
        interactions_person_df = interactions_indexed_df \
            .loc[interactions_indexed_df['user_id'] \
                 == person_id]
        ids = interactions_person_df['click_article_id'].values[0].tolist()
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
    user_profiles = build_users_profiles(clicks)
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
    global ds
    global usr_ds
    embeddings_path, metadata_path, matrix_size, clicks = load_params()
    # Filtrage des utilisateurs clickant sur des articles référencés en base de données
    user = choice(usr_ds.loc[usr_ds['click_article_id'].isin(ds['article_id'])]['user_id'].values.tolist())
    print(user)  # Affichage de l’utilisateur sélectionné
    recomm = similar_items_to_user_profile(user, topn=5)  # Retour des 5 meilleures recommandations
    print(recomm)
    with open('src/inference_content_based_filter.txt', 'w') as f:
        f.write('user : ' + str(user) + '\n' + 'recommendation : ' + str(recomm))


if __name__ == "__main__":
    main()
