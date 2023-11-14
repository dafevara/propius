
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




class SimilarityModel:
    """
    SimilarityModel is the training interface to extract similar items in a dataset
    """

    def __init__(
        self,
        dictionary: pd.DataFrame,
        item_occurrences: pd.DataFrame,
        occurrences_size: int
    ):
        """
        Arguments:
            dictionay: a dataframe with all unique items
            item_occurrences: a dataframe with all item co-occurrences
            occurrences_size: the size of item_occurrences must be know up front
                in order to pre-allocate memory space making calculations faster and lighter.
        """
        self.dictionary = dictionary
        self.item_occurrences = item_occurrences
        self.occurrences_size = occurrences_size
        self.__df_corr: pd.DataFrame = pd.DataFrame()
        self.__csr_matrix: sparse.csr.csr_matrix = None
        self.__corr_coeffs: pd.DataFrame = pd.DataFrame()

    def __correlation_coefficients(self, A, B=None) -> np.matrix:
        if B is not None:
            A = sparse.vstack((A, B), format='csr')

        A = A.astype(np.float32)
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

    def __crosstab(self) -> sparse.csr.csr_matrix:
        rows_pos = np.empty(self.occurrences_size, dtype=np.int32)
        col_pos = np.empty(self.occurrences_size, dtype=np.int32)
        data = np.empty(self.occurrences_size, dtype=np.int32)

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
        """
        Creates a CSR Matrix to represent items co-occurences. Each column is
        treated as an independent variable, so they can be correlated among each
        other. Each correlation will be used as the way to define similarity
        among them.
        """
        self.__csr_matrix = self.__crosstab()
        self.__corr_coeffs = self.__correlation_coefficients(
            self.__csr_matrix
        )

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return:
            Dataframe which stores items correlations
        """
        if self.__corr_coeffs is None:
            return

        if self.__df_corr.empty:
            self.__df_corr = pd.DataFrame(self.__corr_coeffs)

        return self.__df_corr

    def save_csr_matrix(self, output_dir='/tmp') -> str:
        """
        Saves the CSR Matrix as a npz file in the _output_dir_
        Arguments:
            output_dir: npz file destionation dir
        Return:
            CSR Matrix file path
        """
        file_path = f'{output_dir}/csr_matrix.npz'
        sparse.save_npz(file_path)

        return file_path

    def save_correlation_coeffs(self, output_dir='/tmp') -> str:
        """
        Saves correlations dataframe as a npz file in the _output_dir_
        Arguments:
            output_dir: npz file destionation dir
        Return:
            Correlations dataframe file path
        """
        file_path = f'{output_dir}/corr_coeff_matrix.npz'
        sparse.save_npz(file_path)

        return file_path

    def store_in_db(self):
        """
        Stores all item correlations (similarities) in a local db (sqlite3).
        Local db path can be specified in the config file.
        """

        from persistance import ModelStorer

        storer = ModelStorer(self)
        storer.prepare()
        storer.populate_correlated_items()
        storer.populate_similar_items()
