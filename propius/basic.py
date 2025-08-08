import utils
import similarity
import data
import pandas as pd

def main():
    """Main example demonstrating the functional similarity calculation approach."""
    
    # Load co-occurrence data
    item_occurrences = utils.stream_csv('/tmp/imp_ordered_person_titles.csv')
    print(f"Loaded item occurrences: {item_occurrences}")
    
    # Build similarity matrix using functional approach
    correlation_matrix = similarity.build_similarity_matrix(
        occurrences_data=item_occurrences,
        occurrences_size=2399055
    )
    
    # Convert to DataFrame for easier manipulation
    correlation_df = similarity.correlation_matrix_to_dataframe(correlation_matrix)
    print(f"Correlation matrix shape: {correlation_df.shape}")
    print(correlation_df.head(20))
    
    # Example: Get similar items for selected items
    try:
        for i in [10, 187, 63]:
            similar_items = similarity.get_similar_items(
                correlation_df=correlation_df,
                item_id=i,
                threshold_method="std_dev",
                threshold_value=2.0
            )
            print(f"Similar items to item {i}:")
            print(similar_items.head())
    except ValueError as e:
        print(f"Could not find similar items: {e}")
    
if __name__ == "__main__":
    main()
