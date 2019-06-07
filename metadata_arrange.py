import os

import numpy as np
import pandas as pd

meta_path = os.path.abspath("item_metadata.csv")
meta = pd.read_csv(meta_path)

pro = pd.DataFrame(meta.properties)
item = pd.DataFrame(meta.item_id)

split = pro.properties.str.split('|')
split_ = split.apply(lambda x : pd.Series(x))

split_sin = split_.stack().reset_index(level=1,drop=True).to_frame('property_single')
property_single = split_sin['property_single'].unique()

pro_one_hot_encoded = pd.get_dummies(split_sin.property_single)
for x in range(item.shape[0]):
    if x == 0:
        p = pro_one_hot_encoded.loc[x].sum(axis = 0).to_frame().T
        pro_one_hot_encoded_sum = p
    else:
        p = pro_one_hot_encoded.loc[x].sum(axis = 0).to_frame().T
        pd.concat([pro_one_hot_encoded_sum, p], axis = 0)
