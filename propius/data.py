"""
Data storage and persistence functions for similarity results.

This module handles storing and retrieving similarity data from databases,
providing a clean separation between similarity calculations and data persistence.

This module consolidates and replaces the old persistance.py module, providing
both functional and backward-compatible interfaces.
"""

import pandas as pd
import numpy as np
import sqlite3 as db
from typing import Optional, Union
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def store_similarities_in_database(correlation_df: pd.DataFrame, 
                                 dictionary: pd.DataFrame,
                                 db_path: str = '/tmp/propius.db') -> None:
    """
    Store similarity results in a SQLite database.
    
    Args:
        correlation_df: DataFrame with correlation coefficients
        dictionary: DataFrame with item mappings (must have 'title' column)
        db_path: Path to SQLite database file
    """
    print(f"Storing similarities in database: {db_path}")
    
    # Prepare database tables
    prepare_database_tables(db_path)
    
    # Populate correlated items table
    populate_correlated_items(dictionary, db_path)
    
    # Populate similar items table
    populate_similar_items(correlation_df, db_path)
    
    print(f"Successfully stored similarities in database: {db_path}")


def prepare_database_tables(db_path: str) -> None:
    """
    Prepare database tables for storing similarities.
    
    Creates the necessary tables and indexes for storing item correlations
    and similarity scores.
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = db.connect(db_path)
    cur = conn.cursor()
    
    # Drop and create correlated_items table
    cur.execute('DROP TABLE IF EXISTS correlated_items;')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS correlated_items(
            id  INTEGER PRIMARY KEY,
            key   TEXT,
            human_label TEXT
        );
    ''')
    cur.execute('DROP INDEX IF EXISTS ux__correlated_items__key;')
    cur.execute('''
        CREATE UNIQUE INDEX ux__correlated_items__key
        on correlated_items (key);
    ''')
    
    # Drop and create similar_items table
    cur.execute('DROP TABLE IF EXISTS similar_items;')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS similar_items(
            item_a_id      INTEGER,
            item_b_id      INTEGER,
            scaled_score   REAL
        );
    ''')
    cur.execute('DROP INDEX IF EXISTS ux__similar_items__item_a_id__item_b_id;')
    cur.execute('''
        CREATE UNIQUE INDEX ux__similar_items__item_a_id__item_b_id
        on similar_items (item_a_id, item_b_id);
    ''')
    
    conn.close()


def populate_correlated_items(dictionary: pd.DataFrame, db_path: str) -> None:
    """
    Populate the correlated_items table with item dictionary data.
    
    Args:
        dictionary: DataFrame with item mappings (must have 'title' column)
        db_path: Path to SQLite database file
    """
    conn = db.connect(db_path)
    cur = conn.cursor()
    
    # Generator for correlated items
    def yield_correlated_items():
        with tqdm(total=dictionary.shape[0], desc="Storing correlated items") as pb:
            for row in dictionary.itertuples():
                pb.update(1)
                yield (row.Index, row.title)
    
    cur.executemany(
        "INSERT INTO correlated_items(id, key) VALUES (?, ?)",
        yield_correlated_items()
    )
    conn.commit()
    conn.close()


def populate_similar_items(correlation_df: pd.DataFrame, 
                         db_path: str,
                         threshold_std_multiplier: float = 2.0) -> None:
    """
    Populate the similar_items table with similarity scores.
    
    Only stores items that have similarity scores above a statistical threshold
    (mean + std * threshold_std_multiplier).
    
    Args:
        correlation_df: DataFrame with correlation coefficients
        db_path: Path to SQLite database file
        threshold_std_multiplier: Standard deviation multiplier for threshold
    """
    with tqdm(total=len(correlation_df.index), desc="Storing similar items") as pb:
        for item_id in correlation_df.index:
            similars = correlation_df.loc[:, item_id].reset_index()
            similars = similars[similars['index'] != item_id]

            # Scale similarities using MinMaxScaler
            scaler = MinMaxScaler()
            scaled_similars = similars.copy()
            scaled_similars[[item_id]] = scaler.fit_transform(scaled_similars[[item_id]])

            # Apply statistical threshold (mean + std * multiplier)
            std = scaled_similars[item_id].std()
            mean = scaled_similars[item_id].mean()
            cut_off = mean + (std * threshold_std_multiplier)

            filtered = scaled_similars[scaled_similars[item_id] >= cut_off]
            filtered = filtered.sort_values(item_id, ascending=False)
            
            insert_item_similarities(item_id, filtered, db_path)
            pb.update(1)


def insert_item_similarities(item_id: int, 
                           similars: pd.DataFrame, 
                           db_path: str) -> None:
    """
    Insert similarity scores for a specific item into the database.
    
    Args:
        item_id: ID of the primary item
        similars: DataFrame with similar items and their scores
        db_path: Path to SQLite database file
    """
    if similars.empty:
        return
        
    # Prepare condition for SQL query
    condition = np.append(similars['index'].values, item_id)
    if len(condition) <= 1:
        condition = f'({condition[0]})'
    else:
        condition = f'{tuple(condition)}'
    
    query = f'''
        SELECT id, key
        FROM correlated_items
        WHERE id IN {condition}
    '''
    
    conn = db.connect(db_path)
    similar_titles_df = pd.read_sql_query(query, conn)

    df = similar_titles_df \
        .set_index('id') \
        .join(similars.set_index('index')) \
        .sort_values(item_id, ascending=False)

    df = df.drop(item_id)
    df.rename(columns={item_id: 'scaled_score'}, inplace=True)
    
    # Prepare data for insertion
    data = []
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
                INSERT INTO similar_items (item_a_id, item_b_id, scaled_score)
                VALUES (?, ?, ?)
            ''',
            data_gen
        )
        conn.commit()

    conn.close()


def retrieve_similar_items(item_id: int, 
                         db_path: str = '/tmp/propius.db',
                         limit: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieve similar items for a given item from the database.
    
    Args:
        item_id: ID of the item to find similarities for
        db_path: Path to SQLite database file
        limit: Maximum number of similar items to return
        
    Returns:
        DataFrame with similar items and their scores
    """
    conn = db.connect(db_path)
    
    query = '''
        SELECT 
            si.item_b_id,
            ci.key as item_name,
            si.scaled_score
        FROM similar_items si
        JOIN correlated_items ci ON si.item_b_id = ci.id
        WHERE si.item_a_id = ?
        ORDER BY si.scaled_score DESC
    '''
    
    if limit:
        query += f' LIMIT {limit}'
    
    result = pd.read_sql_query(query, conn, params=(item_id,))
    conn.close()
    
    return result


def get_item_info(item_id: int, db_path: str = '/tmp/propius.db') -> Optional[dict]:
    """
    Get information about a specific item from the database.
    
    Args:
        item_id: ID of the item
        db_path: Path to SQLite database file
        
    Returns:
        Dictionary with item information or None if not found
    """
    conn = db.connect(db_path)
    
    query = '''
        SELECT id, key, human_label
        FROM correlated_items
        WHERE id = ?
    '''
    
    result = pd.read_sql_query(query, conn, params=(item_id,))
    conn.close()
    
    if result.empty:
        return None
    
    return result.iloc[0].to_dict()


def search_items_by_name(search_term: str, 
                        db_path: str = '/tmp/propius.db',
                        limit: int = 10) -> pd.DataFrame:
    """
    Search for items by name in the database.
    
    Args:
        search_term: Search term to look for in item names
        db_path: Path to SQLite database file
        limit: Maximum number of results to return
        
    Returns:
        DataFrame with matching items
    """
    conn = db.connect(db_path)
    
    query = '''
        SELECT id, key, human_label
        FROM correlated_items
        WHERE key LIKE ? OR human_label LIKE ?
        ORDER BY key
        LIMIT ?
    '''
    
    search_pattern = f'%{search_term}%'
    result = pd.read_sql_query(
        query, 
        conn, 
        params=(search_pattern, search_pattern, limit)
    )
    conn.close()
    
    return result


def get_database_stats(db_path: str = '/tmp/propius.db') -> dict:
    """
    Get statistics about the stored data.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Dictionary with database statistics
    """
    conn = db.connect(db_path)
    
    # Get item count
    item_count = pd.read_sql_query(
        'SELECT COUNT(*) as count FROM correlated_items', 
        conn
    ).iloc[0]['count']
    
    # Get similarity count
    similarity_count = pd.read_sql_query(
        'SELECT COUNT(*) as count FROM similar_items', 
        conn
    ).iloc[0]['count']
    
    # Get average similarities per item
    avg_similarities = pd.read_sql_query(
        '''
        SELECT AVG(similarity_count) as avg_count
        FROM (
            SELECT item_a_id, COUNT(*) as similarity_count
            FROM similar_items
            GROUP BY item_a_id
        )
        ''', 
        conn
    ).iloc[0]['avg_count']
    
    conn.close()
    
    return {
        'total_items': item_count,
        'total_similarities': similarity_count,
        'avg_similarities_per_item': round(avg_similarities, 2) if avg_similarities else 0
    }


# Backward compatibility functions to replace ModelStorer class
def store_similarity_model_in_database(similarity_model, 
                                     dictionary: pd.DataFrame,
                                     db_path: str = '/tmp/propius.db') -> None:
    """
    Store similarity model results in database (backward compatibility function).
    
    This function provides backward compatibility with the old ModelStorer class
    by accepting a SimilarityModel object and extracting its DataFrame.
    
    Args:
        similarity_model: SimilarityModel object (or any object with as_dataframe() method)
        dictionary: DataFrame with item mappings
        db_path: Path to SQLite database file
    """
    # Extract DataFrame from similarity model
    if hasattr(similarity_model, 'as_dataframe'):
        correlation_df = similarity_model.as_dataframe()
    else:
        raise ValueError("similarity_model must have an 'as_dataframe()' method")
    
    # Use the consolidated functional approach
    store_similarities_in_database(correlation_df, dictionary, db_path)


class ModelStorer:
    """
    Backward compatibility class that wraps the functional approach.
    
    This class provides the same interface as the old ModelStorer but internally
    uses the new functional approach. This allows existing code to work without
    changes while encouraging migration to the functional approach.
    
    Deprecated: Use store_similarities_in_database() function instead.
    """
    
    def __init__(self, similarity_model, dictionary: pd.DataFrame, db_path: str = '/tmp/propius.db'):
        """
        Initialize ModelStorer with backward compatibility.
        
        Args:
            similarity_model: SimilarityModel object or DataFrame
            dictionary: DataFrame with item mappings
            db_path: Path to SQLite database file
        """
        self.similarity_model = similarity_model
        self.dictionary = dictionary
        self._db_path = db_path
        
        # Warning about deprecation
        import warnings
        warnings.warn(
            "ModelStorer class is deprecated. Use store_similarities_in_database() function instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def prepare(self) -> None:
        """Prepare database tables (backward compatibility method)."""
        prepare_database_tables(self._db_path)
    
    def populate_correlated_items(self) -> None:
        """Populate correlated items table (backward compatibility method)."""
        populate_correlated_items(self.dictionary, self._db_path)
    
    def populate_similar_items(self) -> None:
        """Populate similar items table (backward compatibility method)."""
        # Extract DataFrame from similarity model
        if hasattr(self.similarity_model, 'as_dataframe'):
            correlation_df = self.similarity_model.as_dataframe()
        elif isinstance(self.similarity_model, pd.DataFrame):
            correlation_df = self.similarity_model
        else:
            raise ValueError("similarity_model must be a DataFrame or have an 'as_dataframe()' method")
        
        populate_similar_items(correlation_df, self._db_path)


# Legacy function aliases for backward compatibility
def prepare_model_storage(db_path: str = '/tmp/propius.db') -> None:
    """Legacy alias for prepare_database_tables() - deprecated."""
    import warnings
    warnings.warn(
        "prepare_model_storage() is deprecated. Use prepare_database_tables() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    prepare_database_tables(db_path)


def store_model_similarities(similarity_model, 
                           dictionary: pd.DataFrame,
                           db_path: str = '/tmp/propius.db') -> None:
    """Legacy function for storing model similarities - deprecated."""
    import warnings
    warnings.warn(
        "store_model_similarities() is deprecated. Use store_similarities_in_database() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    store_similarity_model_in_database(similarity_model, dictionary, db_path)
