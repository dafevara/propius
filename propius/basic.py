from utils import stream_csv
from model import SimilarityModel

import pandas as pd

dictionary = pd.read_csv('/tmp/dictionary.csv')
item_occurrences = stream_csv('/tmp/ordered_person_titles.csv')

model = SimilarityModel(
    item_occurrences,
    2739984
)

model.build()
df = model.as_dataframe()
