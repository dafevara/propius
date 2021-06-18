import utils
import model

import pandas as pd

dictionary = pd.read_csv(
    '/.../dictionary.csv'
)
item_occurrences = utils.stream_csv(
    '/.../item_co_occurrences.csv'
)

sim_model = model.SimilarityModel(
    dictionary,
    item_occurrences,
    1_234_567  # num of occurrences must be previously known
)

sim_model.build()
df = sim_model.as_dataframe()
sim_model.store_in_db()
