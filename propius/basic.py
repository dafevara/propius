import utils
import model

import pandas as pd

dictionary = pd.read_csv(
    '/Users/dafevara/PROJECTS/propius/examples/dictionary.csv'
)
item_occurrences = utils.stream_csv(
    '/Users/dafevara/PROJECTS/propius/examples/imp_ordered_person_titles.csv'
)

sim_model = model.SimilarityModel(
    dictionary,
    item_occurrences,
    2399055  # num of occurrences must be previously known
)

sim_model.build()
df = sim_model.as_dataframe()
sim_model.store_in_db()
