
import pandas as pd
import numpy as np
import sqlite3 as db

from typing import Any
from typing import TypeVar

from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

from model import SimilarityModel


class ModelStorer():
    """
    Since Propius allow to extract similar items in big data volumes sometimes
    it takes several hours and significant amount of memory, disk space and cpu to
    uncover those similarities when it comes to thousands of different items or millions
    of different co-occurrences of those thousands of items. So, to avoid investing
    time and effort each time we need those similarities, Propius provides a interface
    to store the similarities in a local db (sqlite3) after each training process so
    the similarities can be retrieved from it.
    """
    def __init__(self, similarity_model: SimilarityModel):
        self.similarity_model = similarity_model
        self.__db_path: str = '/tmp/propius.db'

    def prepare(self):
        """
        Prepares tables to store similarities and found correlated items
        """

        conn = db.connect(self.__db_path)
        cur = conn.cursor()
        cur.execute('''
            DROP TABLE IF EXISTS correlated_items;
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS correlated_items(
                id  SERIAL PRIMARY KEY,
                key   text,
                human_label text
            );
        ''')
        cur.execute('''
            DROP INDEX IF EXISTS ux__correlated_items__key;
        ''')
        cur.execute('''
            CREATE UNIQUE INDEX ux__correlated_items__key
            on correlated_items (key);
        ''')
        cur.execute('''
            DROP TABLE IF EXISTS similar_items;
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS similar_items(
                item_a_id      integer,
                item_b_id      integer,
                scaled_score    float
            );
        ''')
        cur.execute('''
            DROP INDEX IF EXISTS ux__similar_items__item_a_id__item_b_id;
        ''')
        cur.execute('''
            CREATE UNIQUE INDEX ux__similar_items__item_a_id__item_b_id
            on similar_items (item_a_id, item_b_id);
        ''')

        conn.close()

    def __yield_correlated_items(self):
        df_item_dict = self.similarity_model.dictionary
        with tqdm(total=df_item_dict.shape[0]) as pb:
            for row in df_item_dict.itertuples():
                pb.update(1)
                yield (row.Index, row.title,)

    def populate_correlated_items(self):
        """
        Saves all found correlated items in the _correlated_items_ table.
        """
        conn = db.connect(self.__db_path)
        cur = conn.cursor()

        cur.executemany(
            "insert into correlated_items(id, key) values (?, ?)",
            self.__yield_correlated_items()
        )
        conn.commit()
        conn.close()

    def populate_similar_items(self):
        """
        Saves all similar items per each item which its correlation value is at least
        `mean + std*2`
        """
        scaler = MinMaxScaler()
        df_corr = self.similarity_model.as_dataframe()
        with tqdm(total=len(df_corr.index)) as pb:
            for item_id in df_corr.index:
                similars = df_corr.loc[:, item_id].reset_index()
                similars = similars[similars['index'] != item_id]

                scaled_similars = similars.copy()
                scaled_similars[[item_id]] = scaler.fit_transform(scaled_similars[[item_id]])

                std = scaled_similars[item_id].std()
                mean = scaled_similars[item_id].mean()
                cut_off = mean + (std * 2)

                filtered = scaled_similars[scaled_similars[item_id] >= cut_off]
                filtered = filtered.sort_values(item_id, ascending=False)
                self.__insert_similarities(item_id, filtered)
                pb.update(1)

    def __insert_similarities(self, item_id: int, similars: pd.DataFrame):
        condition = np.append(similars['index'].values, item_id)
        if len(condition) <= 1:
            condition = f'({condition[0]})'
        else:
            condition = f'{tuple(condition)}'
        query = f'''
            select id, key
            from correlated_items
            where id in {condition}
        '''
        data = []
        conn = db.connect(self.__db_path)
        similar_titles_df = pd.read_sql_query(query, conn)

        df = similar_titles_df \
            .set_index('id') \
            .join(similars.set_index('index')) \
            .sort_values(item_id, ascending=False)

        df = df.drop(item_id)
        df.rename(columns={item_id: 'scaled_score'}, inplace=True)
        for tt in df.reset_index().itertuples():
            data.append({
                'item_a_id': item_id,
                'item_b_id': tt.id,
                'scaled_score': tt.scaled_score
            })

        if data:
            cur = conn.cursor()
            data_gen = (
                (row['item_a_id'], row['item_b_id'], row['scaled_score'])
                for row in data
            )
            cur.executemany(
                '''
                    insert into similar_items (item_a_id, item_b_id, scaled_score)
                    values (?, ?, ?)
                ''',
                data_gen
            )
            conn.commit()

        conn.close()
