import os

import numpy as np
import pandas as pd

meta_path = os.path.abspath("item_metadata.csv")
meta = pd.read_csv(meta_path)

# Data split (item / properties)
pro = pd.DataFrame(meta.properties)
item = pd.DataFrame(meta.item_id)

# properties split each
split = pro.properties.str.split('|')
split_ = split.apply(lambda x : pd.Series(x))

# arrange properties for each index
split_sin = split_.stack().reset_index(level=1,drop=True).to_frame('property_single')
property_single = split_sin['property_single'].unique()

# make one hot vector about all properties
pro_one_hot_encoded = pd.get_dummies(split_sin.property_single)

# sum one hot vector for each index
for x in range(item.shape[0]):
    if x == 0:
        p = pro_one_hot_encoded.loc[x].sum(axis = 0).to_frame().T
        pro_one_hot_encoded_sum = p
    elif pro_one_hot_encoded.loc[x].size == 157:
        p = pro_one_hot_encoded.loc[x].to_frame().T
        pd.concat([pro_one_hot_encoded_sum,p], axis = 0)
    else:
        p = pro_one_hot_encoded.loc[x].sum(axis = 0).to_frame().T
        pd.concat([pro_one_hot_encoded_sum, p], axis = 0)
