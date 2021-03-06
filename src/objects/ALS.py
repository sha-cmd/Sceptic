"""Ce fichier crée une base de données de recommendation pour nos clients selon nos articles"""

import argparse
import codecs
import logging
import time

import numpy as np
import tqdm
import pandas as pd
import scipy

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)

# Tous ces modèles sont des systèmes de recommandation de collaboration filtering, mais avec des modalités
# différentes, nuançant nos résultats selon le taux d’engagement par nos clients pour le profit.
MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "faiss_als": FaissAlternatingLeastSquares,
    "tfidf": TFIDFRecommender,
    "cosine": CosineRecommender,
    "bpr": BayesianPersonalizedRanking,
    "lmf": LogisticMatrixFactorization,
    "bm25": BM25Recommender,
}


def get_model(model_name):
    """Accesseur du modèle, dans un mode fonctionnel, adaptable à un objet"""
    print(f"getting model {model_name}")
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError(f"Unknown Model '{model_name}'")

    if model_name.endswith("als"):
        params = {"factors": 128, "dtype": np.float32}
    elif model_name == "bm25":
        params = {"K1": 100, "B": 0.5}
    elif model_name == "bpr":
        params = {"factors": 63}
    elif model_name == "lmf":
        params = {"factors": 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)


def calculate_similar_books(output_filename, model_name="als"):
    """Computation des livres similaires dans notre base de données"""
    df = pd.read_csv('data/database.csv', sep=';', index_col='index')[['itemID', 'userID', 'rating']]
    books = {i: x for i, x in enumerate(df['itemID'].unique())}
    plays = df[['userID', 'itemID', 'rating']]
    model = get_model(model_name)

    if model_name.endswith("als"):
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        model.approximate_recommend = False

    # Création d’un index de récupération de nos sujets d’observation
    item = {x: 0 for x in df['itemID'].unique()}
    user = {x: {} for x in df['userID'].unique()}
    for i in range(len(df)):
        item[df.iloc[i, 0]] = df.iloc[i, 2]
        user[df.iloc[i, 1]][df.iloc[i, 0]] = df.iloc[i, 2]
    # Utilisation de la mémoire en mode matricielle et clairsemé, optimisation computationnel, ne pas toucher
    user_plays = scipy.sparse.csr_matrix(pd.DataFrame.from_dict(user).fillna(0).T.values)

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(user_plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    start = time.time()
    logging.debug("calculating top books")

    user_count = df['itemID'].value_counts().index.tolist()
    to_generate = sorted(np.arange(len(books)), key=lambda x: -user_count[x])

    logging.debug("writing similar items")
    # Production du calcul de sortie, inscription dans un fichier de type tableur
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            batch_size = 1000
            for startidx in range(0, len(to_generate), batch_size):
                batch = to_generate[startidx: startidx + batch_size]
                ids, scores = model.similar_items(batch, 11)
                for i, bookid in enumerate(batch):
                    book = books[bookid]
                    if i < 10:
                        for other, score in zip(ids[i], scores[i]):
                            o.write(f"{book}\t{books[other]}\t{score}\n")
                progress.update(batch_size)

    logging.debug("generated similar books in %0.2fs", time.time() - start)


def calculate_recommendations(output_filename, model_name="als"):
    """Calcul des recommandations pour l’ensemble de la base des utilisateurs"""
    df = pd.read_csv('data/database.csv', sep=';', index_col='index')[['itemID', 'userID', 'rating']]
    users = {i: x for i, x in enumerate(df['userID'].unique())}
    books = {i: x for i, x in enumerate(df['itemID'].unique())}
    plays = df[['userID', 'itemID', 'rating']]
    model = get_model(model_name)

    if model_name.endswith("als"):
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        model.approximate_similar_items = False


    item = {x: 0 for x in df['itemID'].unique()}
    user = {x: {} for x in df['userID'].unique()}
    for i in range(len(df)):
        item[df.iloc[i, 0]] = df.iloc[i, 2]
        user[df.iloc[i, 1]][df.iloc[i, 0]] = df.iloc[i, 2]
    # Optimisation
    user_plays = scipy.sparse.csr_matrix(pd.DataFrame.from_dict(user).fillna(0).T.values)


    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(user_plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    start = time.time()
    # Coeur de la computation
    with tqdm.tqdm(total=len(users)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            batch_size = 1000
            to_generate = np.arange(len(users))
            for startidx in range(0, len(to_generate), batch_size):
                batch = to_generate[startidx : startidx + batch_size]
                ids, scores = model.recommend(
                    batch, user_plays[batch], filter_already_liked_items=True
                )
                for i, userid in enumerate(batch):
                    username = users[userid]
                    for other, score in zip(ids[i], scores[i]):
                        if not other == -1:
                            o.write(f"{username}\t{books[other]}\t{score}\n")
                progress.update(batch_size)
    logging.debug("generated recommendations in %0.2fs", time.time() - start)


if __name__ == "__main__":
    # Production d’un appel avec passage d’argument comprenant une valeur par défaut.
    parser = argparse.ArgumentParser(
        description="Generates similar books on the last.fm dataset"
        " or generates personalized recommendations for each user",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/similar-books.tsv",
        dest="outputfile",
        help="output file name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="als",
        dest="model",
        help=f"model to calculate ({'/'.join(MODELS.keys())})",
    )
    parser.add_argument(
        "--recommend",
        help="Recommend items for each user rather than calculate similar_items",
        action="store_true",
    )
    parser.add_argument(
        "--param", action="append", help="Parameters to pass to the model, formatted as 'KEY=VALUE"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.recommend:
        calculate_recommendations(args.outputfile, model_name=args.model)
    else:
        calculate_similar_books(args.outputfile, model_name=args.model)