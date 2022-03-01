import numpy as np
import pandas as pd
import joblib

global model
global ds
global matrix_size


def predict(idx):
    global ds
    for idx, row in ds.loc[ds['article_id'] == idx][['article_id', 'category_id', 'words_count']].iterrows():
        description = f"Article {row['article_id']}, Category {row['category_id']}, Mots {row['words_count']}"
    return description


def recommend(item_id, num):
    global model
    description = str()
    description = description + ("Recommending " + str(num) + " products similar to " + predict(item_id) + "...\n")
    description = description + ("-------\n")
    recs = model[item_id][:num]
    for rec in recs:
        description = description + ("Recommended: " + predict(rec[1]) + " (score:" + str(rec[0]) + ")\n")
    return description


def main():
    global model
    global ds
    global matrix_size
    items = 4
    nb_recommended = 10
    matrix_size = 10000
    ds = pd.read_csv('/home/romain/Documents/BackUp/Special/Projets/Code_IIA/Projet_9/data/articles_metadata.csv').iloc[
         :matrix_size, :]
    model = joblib.load('src/model.joblib')
    inference = recommend(items, nb_recommended)
    with open('src/inference_content_based_filter.txt', 'w') as f:
        f.writelines(inference)


if __name__ == "__main__":
    main()
