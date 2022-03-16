import pandas as pd
import yaml

from surprise import Dataset
from surprise import Reader


class BookLib:

    def __init__(self):
        print('Creation of BookLib Object')
        self.data = None
        self.name_to_bookID = None
        self.bookID_to_name = None
        self.userID_to_name = None
        self.name_to_userID = None
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
            self.book_db = str(params['data']['ranking'])

    def __call__(self):
        print('Chargement des données')
        self.data = pd.read_csv(self.book_db, index_col='index', sep=';')  # Dev mode
        self.name_to_userID, self.userID_to_name = self.raw_and_ids("userID")
        self.name_to_bookID, self.bookID_to_name = self.raw_and_ids("itemID")
        reader = Reader(rating_scale=(1, 5))
        ratingsDataset = Dataset.load_from_df(self.data[['userID', 'itemID', 'rating']], reader)
        return ratingsDataset

    def raw_and_ids(self, name):
        raw_ids = self.data[name].unique().tolist()
        raw_to_innerID = {x: i for i, x in enumerate(raw_ids)}
        innerID_to_raw = {i: x for i, x in enumerate(raw_ids)}
        return raw_to_innerID, innerID_to_raw
