import numpy as np
import pandas as pd

origin_user_df = pd.read_csv('./users.dat', sep='::',names=['user_id', 'gender', 'age', 'occupation', 'zipcode'], header=None)
origin_user_df['user_id'], unique_ids = pd.factorize(origin_user_df['user_id'])
user_id_mapping = {id: i for i, id in enumerate(unique_ids)}

origin_item_df = pd.read_csv('./movies.dat', sep='::',names=['item_id', 'title', 'genres'], header=None, encoding="unicode_escape")
origin_item_df['item_id'], unique_ids = pd.factorize(origin_item_df['item_id'])
item_id_mapping = {id: i for i, id in enumerate(unique_ids)}

ratings_df = pd.read_csv('./ratings.dat', sep='::',header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])

ratings_df['user_id'] = ratings_df['user_id'].map(user_id_mapping)
ratings_df['movie_id'] = ratings_df['movie_id'].map(item_id_mapping)
ratings_df = ratings_df.rename(columns={'user_id': 'user', 'movie_id': 'item', })

ratings_df.to_csv('interaction.csv', index=False)
origin_user_df.to_csv('./users.csv', index=False)
origin_item_df.to_csv('./items.csv', index=False)