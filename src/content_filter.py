import pandas as pd
import numpy as np
import yaml
from sklearn.metrics.pairwise import linear_kernel
import joblib
global ds  # Valeur Initialisée dans cos_sim
global model  # Tableau de notre modèle, sera sauvé sur disque


def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        embeddings_path = str(params['data']['embeddings'])
        metadata_path = str(params['data']['metadata'])
        matrix_size = int(params['content_filter']['matrix_size'])
    print(embeddings_path)
    print('chargement du fichier embeddings')

    return embeddings_path, metadata_path, matrix_size


def cos_sim(embeddings_path, metadata_path, matrix_size):
    global ds
    global model
    ds = pd.read_csv(metadata_path).iloc[:matrix_size, :]
    tfidf_matrix = pd.read_pickle(embeddings_path)[:matrix_size, :]
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    results = {}
    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['article_id'][i]) for i in similar_indices]
        results[row['article_id']] = similar_items[1:]
    model = results
    joblib.dump(model, 'src/model.joblib')


def predict(idx):
    global ds
    for idx, row in ds.loc[ds['article_id'] == idx][['article_id', 'category_id', 'words_count']].iterrows():
        description = f"Article {row['article_id']}, Category {row['category_id']}, Mots {row['words_count']}"
    return description


def recommend(item_id, num):
    global model
    print("Recommending " + str(num) + " products similar to " + predict(item_id) + "...")
    print("-------")
    recs = model[item_id][:num]
    for rec in recs:
        print("Recommended: " + predict(rec[1]) + " (score:" + str(rec[0]) + ")")


def main():
    global model
    embeddings_path, metadata_path, matrix_size = load_params()
    cos_sim(embeddings_path, metadata_path, matrix_size)
    recommend(5, 15)

if __name__ == "__main__":
    main()
