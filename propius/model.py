from typing import Any
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import sqlite3 as db


@dataclass
class SimilarityModel:
    dictionary: pd.DataFrame
    item_occurrences: pd.DataFrame
    occurrences_size: int

    __df_corr: pd.DataFrame = field(init=False, default=pd.DataFrame())
    __csr_matrix: Any = field(init=False)
    __corr_coeffs: Any = field(init=False)

    def __correlation_coefficients(self, A, B=None):
        if B is not None:
            A = sparse.vstack((A, B), format='csr')

        A = A.astype(np.float64)
        n = A.shape[1]

        # Compute the covariance matrix
        rowsum = A.sum(1)
        centering = rowsum.dot(rowsum.T.conjugate()) / n
        C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

        # The correlation coefficients are given by
        # C_{i,j} / sqrt(C_{i} * C_{j})
        d = np.diag(C)
        coeffs = C / np.sqrt(np.outer(d, d))

        return coeffs

    def __crosstab(self):
        rows_pos = np.empty(self.occurrences_size, dtype=np.int32)
        col_pos = np.empty(self.occurrences_size, dtype=np.int32)
        data = np.empty(self.occurrences_size, dtype=np.int32)
        mapping = []

        with tqdm(total=self.occurrences_size) as pb:
            last_ref_id = None
            refid_serial = 0
            ix_track = 0
            for batch in self.item_occurrences:
                for enum, row in enumerate(batch.itertuples(), ix_track):
                    if last_ref_id != row.reference_id:
                        refid_serial += 1

                    rows_pos[enum] = row.item_id
                    col_pos[enum] = refid_serial
                    data[enum] = row.count

                    last_ref_id = row.reference_id
                    ix_track = enum
                    pb.update(1)

                ix_track += 1

        coo_m = coo_matrix((data, (rows_pos - 1, col_pos - 1)))

        csr_matrix = coo_m.tocsr()

        return csr_matrix

    def build(self):
        self.__csr_matrix = self.__crosstab()
        self.__corr_coeffs = self.__correlation_coefficients(
            self.__csr_matrix
        )

    def as_dataframe(self):
        if self.__corr_coeffs is None:
            return

        if self.__df_corr.empty:
            self.__df_corr = pd.DataFrame(self.__corr_coeffs)

        return self.__df_corr

    def save_csr_matrix(self, output_dir='/tmp'):
        file_path = f'{output_dir}/csr_matrix.npz'
        sparse.save_npz(file_path)

        return file_path

    def save_correlation_coeffs(self, output_dir='/tmp'):
        file_path = f'{output_dir}/corr_coeff_matrix.npz'
        sparse.save_npz(file_path)

        return file_path

    def store_in_db(self):
        storer = ModelStorer(self)
        storer.prepare()


@dataclass
class ModelStorer:
    similarity_model: SimilarityModel
    __db_path: str = field(default='/tmp/propius.db')

    def prepare(self):
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

    def yield_correlated_items(self):
        df_item_dict = self.similarity_model.dictionary
        with tqdm(total=df_item_dict.shape[0]) as pb:
            for row in df_item_dict.itertuples():
                pb.update(1)
                yield (row.Index, row.title,)

    def populate_correlated_items(self):
        try:
            conn = db.connect(self.__db_path)
            cur = conn.cursor()

            cur.executemany(
                "insert into correlated_items(id, key) values (?, ?)",
                self.yield_correlated_items()
            )
            conn.commit()
            conn.close()
        except Exception as ex:
            print(ex)

        return None

    def populate_similar_items(self):
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
                cut_off = mean + (std*2)

                filtered = scaled_similars[scaled_similars[item_id] >= cut_off]
                filtered = filtered.sort_values(item_id, ascending=False)
                self.insert_similarities(item_id, filtered)
                pb.update(1)

    def insert_similarities(self, item_id, similars):
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
