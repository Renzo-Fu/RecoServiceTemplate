import pickle
from typing import Any, Dict, Tuple


import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from implicit.cpu.als import AlternatingLeastSquares
from rectools.models import LightFMWrapperModel
from lightfm import LightFM

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import dill
import pandas as pd

import os
import pickle

class Pickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Ranker': 
            return Ranker
        return super().find_class(module, name)
    
def load(path:str):
        with open(os.path.join(path), 'rb') as f:
            return Pickler(f).load()



class Ranker:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def recommend_items_to_user(self, test: pd.DataFrame, N_recs: int = 10):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=self.ranker,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
            )

        recs = pd.DataFrame({"user_id": test["user_id"].unique()})
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))
        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        recs = (
        recs[~(recs["user_id"] == recs["sim_user_id"])]
        .merge(self.watched, on=["sim_user_id"], how="left")
        .explode("item_id")
        .sort_values(["user_id", "sim"], ascending=False)
        .drop_duplicates(["user_id", "item_id"], keep="first")
        .merge(self.item_idf, left_on="item_id", right_on="index", how="left")
            )

        recs["score"] = recs["sim"] * recs["idf"]
        recs = recs.sort_values(["user_id", "score"], ascending=False)
        recs["rank"] = recs.groupby("user_id").cumcount() + 1

        cold_start_users = test[~test["user_id"].isin(self.users_inv_mapping)]
        if not cold_start_users.empty and self.cold_start_recommender:
            cold_start_recs = self.cold_start_recommender.recommend_items_to_user(cold_start_users, N_recs)
            recs = pd.concat([recs, cold_start_recs], ignore_index=True)

            return recs[recs["rank"] <= N_recs][["user_id", "item_id", "score", "rank"]]


    def recommendation(self, user_id: int, N_recs: int = 10):
        df = pd.DataFrame({"user_id": [user_id], "item_id": [user_id]})
        return self.predict(df, N_recs=N_recs).item_id.to_list()   



