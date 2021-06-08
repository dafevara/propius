from typing import Any
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy import sparse

import pandas as pd
import numpy as np
import sqlite3 as db


@dataclass
class SimilarityModel:
    item_occurrences: pd.DataFrame
    occurrences_size: int

    __df_corr: pd.DataFrame = field(init=False)
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
        data = []
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

                    last_ref_id = row.reference_id
                    ix_track = enum
                    pb.update(1)

                ix_track += 1

        data = np.repeat(1, len(rows_pos))
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

        return pd.DataFrame(self.__corr_coeffs)

    def save_csr_matrix(self, output_dir='/tmp'):
        file_path = f'{output_dir}/csr_matrix.npz'
        sparse.save_npz(file_path)

        return file_path

    def save_correlation_coeffs(self, output_dir='/tmp'):
        file_path = f'{output_dir}/corr_coeff_matrix.npz'
        sparse.save_npz(file_path)

        return file_path


@dataclass
class ModelStorer:
    similarity_model: SimilarityModel
    dictionary: pd.DataFrame
    __conn: Any = field(init=False)

    def __post_init__(self):
        try:
            self.__conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)

    def prepare(self):
        return None
