import numpy as np
import pandas as pd
import matplotlib as plt
df = pd.read_csv("dengue_features_train3.csv")
df_labels=pd.read_csv("dengue_labels_train.csv")

df.head()