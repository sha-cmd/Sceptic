import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from io import StringIO
from sklearn.model_selection import train_test_split
from objects import RecommenderNet
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


def load_db():
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
    dataframe = pd.read_csv(book_db)
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
    return dataframe


def split_data():
    global user2user_encoded
    global book2book_encoded
    global book_encoded2book
    global user2user_encoded
    global userencoded2user
    global min_rating
    global max_rating
    global num_users
    global num_books
    df = load_db()
    x = df[["user", "book"]].values
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        x, y, test_size=ratio_val_test * 2, random_state=rand_seed, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=rand_seed,
                                                    shuffle=True)
    print('Data : X -> ' + str(x_train.shape[0]))
    return x_train, x_val, x_test, y_train, y_val, y_test, num_users, num_books


def train():
    x_train, x_val, x_test, y_train, y_val, y_test, num_users, num_books = split_data()
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


def predict():
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

    df = load_db()
    model = keras.models.load_model('model')
    book_df = pd.read_csv(metadata_path)

    user_id = df.user_id.sample(1).iloc[0]
    books_watched_by_user = df[df.user_id == user_id]
    books_not_watched = book_df[
        ~book_df["article_id"].isin(books_watched_by_user.click_article_id.values)
    ]["article_id"]
    books_not_watched = list(
        set(books_not_watched).intersection(set(book2book_encoded.keys()))
    )
    books_not_watched = [[book2book_encoded.get(x)] for x in books_not_watched]
    user_encoder = user2user_encoded.get(user_id)
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
        for row in recommended_books.itertuples():
            recomm.append(row.article_id)
        myJSON = [str(x) for x in recomm]

        myArray = StringIO()
        json.dump(myJSON, myArray)
        print(myArray.getvalue())
        return myArray.getvalue()

    except KeyError as e:
        print('Utilisateur sans article concordant avec la matrice tfidf')
        return ('Utilisateur sans article concordant avec la matrice tfidf')


if __name__ == "__main__":
    function_map = {
        'predict': predict,
        'train': train,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    args = parser.parse_args()
    function = function_map[args.command]
    function()

