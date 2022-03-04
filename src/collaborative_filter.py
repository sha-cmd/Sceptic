import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from glob import glob
from random import sample as sampling
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

global ds
global tfidf_matrix
global usr_ds
global usr_ds_ranking
global clicks
global clicks_agg
global clicks_list
global metadata_path
global embeddings_path
global test_ratio
global sample
global epochs
global lr


def load_db():
    global ds
    global tfidf_matrix
    global usr_ds
    global usr_ds_ranking
    global clicks
    global clicks_agg
    global clicks_list
    global metadata_path
    global embeddings_path
    global test_ratio
    global sample
    global epochs
    global lr

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        sample = int(params['collab']['sample'])
        test_ratio = float(params['collab']['ratio'])
        epochs = int(params['collab']['epochs'])
        lr = float(params['collab']['learning_rate'])
        metadata_path = str(params['data']['metadata'])
        book_db = str(params['data']['ranking'])
    index = sampling(pd.read_csv(book_db).index.tolist(), sample)  # Tirage aléatoire sans remise
    df = pd.read_csv(book_db).iloc[index]
    return df

df = load_db()
print('db -> ' + str(df.shape[0]))

user_ids = df["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
book_ids = df["click_article_id"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded2book = {i: x for i, x in enumerate(book_ids)}
df["user"] = df["user_id"].map(user2user_encoded)
df["book"] = df["click_article_id"].map(book2book_encoded)

num_users = len(user2user_encoded)
num_books = len(book_encoded2book)
df["rating"] = df["rating"].values.astype(np.float32)
min_rating = min(df["rating"])
max_rating = max(df["rating"])

description = pd.DataFrame([[num_users, num_books, min_rating, max_rating, sample]], columns=["users", "books", "Min_rating", "max_rating", "lines"])
print(description)

df = df.sample(frac=1, random_state=42)
x = df[["user", "book"]].values
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=test_ratio*2, random_state=42, shuffle=True)
x_val, x_val, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42, shuffle=True)
print('Data : X -> ' + str(x_train.shape[0]))

EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(
            num_books,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=lr)
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)

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
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_book_ids = [
    book_encoded2book.get(books_not_watched[x][0]) for x in top_ratings_indices
]

