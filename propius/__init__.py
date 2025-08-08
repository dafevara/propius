__version__ = "0.1"

from propius import utils
from propius import similarity
from propius import data

from .utils import stream_csv
from .similarity import (
    build_similarity_matrix,
    calculate_correlation_coefficients,
    build_crosstab_matrix,
    correlation_matrix_to_dataframe,
    get_similar_items,
    save_correlation_matrix,
    load_correlation_matrix
)
from .data import (
    store_similarities_in_database,
    retrieve_similar_items,
    get_item_info,
    search_items_by_name,
    get_database_stats,
    ModelStorer,  # Backward compatibility class
    store_similarity_model_in_database
)

__all__ = [
    "utils",
    "similarity",
    "data",
    "stream_csv",
    "build_similarity_matrix",
    "calculate_correlation_coefficients", 
    "build_crosstab_matrix",
    "correlation_matrix_to_dataframe",
    "get_similar_items",
    "save_correlation_matrix",
    "load_correlation_matrix",
    "store_similarities_in_database",
    "retrieve_similar_items",
    "get_item_info",
    "search_items_by_name",
    "get_database_stats",
    "ModelStorer",  # Backward compatibility
    "store_similarity_model_in_database"
]
