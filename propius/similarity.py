"""
Similarity calculation functions for finding correlations between items.

This module provides functional approaches for calculating item similarities
based on co-occurrence data using correlation coefficients and sparse matrices.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse


def calculate_correlation_coefficients(matrix_a: sparse.csr_matrix, 
                                     matrix_b: Optional[sparse.csr_matrix] = None) -> np.matrix:
    """
    Calculate correlation coefficients between items in sparse matrices.
    
    Args:
        matrix_a: Primary sparse CSR matrix
        matrix_b: Optional second matrix to correlate with matrix_a
        
    Returns:
        Correlation coefficients matrix
    """
    if matrix_b is not None:
        matrix_a = sparse.vstack((matrix_a, matrix_b), format='csr')

    matrix_a = matrix_a.astype(np.float32)
    n = matrix_a.shape[1]

    # Compute the covariance matrix
    rowsum = matrix_a.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    covariance = (matrix_a.dot(matrix_a.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    diagonal = np.diag(covariance)
    coeffs = covariance / np.sqrt(np.outer(diagonal, diagonal))

    return coeffs


def build_crosstab_matrix(occurrences_data: Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader],
                         occurrences_size: int) -> sparse.csr_matrix:
    """
    Build a sparse CSR matrix from co-occurrence data.
    
    Creates a cross-tabulation matrix where rows represent items and columns
    represent reference contexts (like users, sessions, etc.).
    
    Args:
        occurrences_data: DataFrame or chunked reader with co-occurrence data
        occurrences_size: Total number of occurrences for memory pre-allocation
        
    Returns:
        Sparse CSR matrix representing item co-occurrences
    """
    # Pre-allocate arrays for better performance
    rows_pos = np.zeros(occurrences_size, dtype=np.int32)
    col_pos = np.zeros(occurrences_size, dtype=np.int32)
    data = np.zeros(occurrences_size, dtype=np.int32)

    with tqdm(total=occurrences_size, desc="Building crosstab matrix") as progress_bar:
        last_ref_id = None
        refid_serial = 0
        ix_track = 0
        
        for batch in occurrences_data:
            for enum, row in enumerate(batch.itertuples(), ix_track):
                if last_ref_id != row.reference_id:
                    refid_serial += 1

                rows_pos[enum] = row.item_id
                col_pos[enum] = refid_serial
                data[enum] = 1.0

                last_ref_id = row.reference_id
                ix_track = enum
                progress_bar.update(1)

            ix_track += 1

    # Create coordinate matrix and convert to CSR format
    coo_matrix_result = coo_matrix((data, (rows_pos - 1, col_pos - 1)))
    return coo_matrix_result.tocsr()


def build_similarity_matrix(occurrences_data: Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader],
                           occurrences_size: int) -> np.matrix:
    """
    Build a complete similarity matrix from co-occurrence data.
    
    This is the main function that combines crosstab creation and correlation
    calculation into a single workflow.
    
    Args:
        occurrences_data: DataFrame or chunked reader with co-occurrence data
        occurrences_size: Total number of occurrences for memory pre-allocation
        
    Returns:
        Correlation coefficients matrix representing item similarities
    """
    print("Building similarity matrix...")
    
    # Step 1: Create sparse matrix from co-occurrences
    crosstab_matrix = build_crosstab_matrix(occurrences_data, occurrences_size)
    
    # Step 2: Calculate correlation coefficients
    print("Calculating correlation coefficients...")
    correlation_matrix = calculate_correlation_coefficients(crosstab_matrix)
    
    print("Similarity matrix build complete.")
    return correlation_matrix


def correlation_matrix_to_dataframe(correlation_matrix: np.matrix) -> pd.DataFrame:
    """
    Convert correlation matrix to pandas DataFrame for easier manipulation.
    
    Args:
        correlation_matrix: Numpy matrix with correlation coefficients
        
    Returns:
        DataFrame representation of the correlation matrix
    """
    return pd.DataFrame(correlation_matrix)


def get_similar_items(correlation_df: pd.DataFrame, 
                     item_id: int, 
                     threshold_method: str = "std_dev",
                     threshold_value: float = 2.0) -> pd.DataFrame:
    """
    Get similar items for a given item based on correlation scores.
    
    Args:
        correlation_df: DataFrame with correlation coefficients
        item_id: ID of the item to find similarities for
        threshold_method: Method to determine similarity threshold ("std_dev" or "absolute")
        threshold_value: Threshold value (multiplier for std_dev or absolute value)
        
    Returns:
        DataFrame with similar items and their similarity scores
    """
    if item_id not in correlation_df.columns:
        raise ValueError(f"Item ID {item_id} not found in correlation matrix")
    
    # Get correlations for the specified item
    item_correlations = correlation_df.loc[:, item_id].reset_index()
    item_correlations = item_correlations[item_correlations['index'] != item_id]
    
    if threshold_method == "std_dev":
        std = item_correlations[item_id].std()
        mean = item_correlations[item_id].mean()
        threshold = mean + (std * threshold_value)
    else:  # absolute threshold
        threshold = threshold_value
    
    # Filter and sort by similarity score
    similar_items = item_correlations[item_correlations[item_id] >= threshold]
    similar_items = similar_items.sort_values(item_id, ascending=False)
    
    return similar_items


def save_correlation_matrix(correlation_matrix: np.matrix, 
                          filepath: str,
                          format: str = "csv") -> None:
    """
    Save correlation matrix to file.
    
    Args:
        correlation_matrix: Matrix to save
        filepath: Path where to save the file
        format: File format ("csv", "numpy", "pickle")
    """
    if format == "csv":
        df = correlation_matrix_to_dataframe(correlation_matrix)
        df.to_csv(filepath, index=False)
    elif format == "numpy":
        np.save(filepath, correlation_matrix)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_correlation_matrix(filepath: str, 
                          format: str = "csv") -> Union[np.matrix, pd.DataFrame]:
    """
    Load correlation matrix from file.
    
    Args:
        filepath: Path to the file
        format: File format ("csv", "numpy")
        
    Returns:
        Loaded correlation matrix
    """
    if format == "csv":
        return pd.read_csv(filepath)
    elif format == "numpy":
        return np.load(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

