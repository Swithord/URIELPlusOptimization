import os
import numpy as np
import pandas as pd

# csv first column is the language name
current_dir = os.path.dirname(__file__)
print("\n path: ", current_dir)
data_path = os.path.join(current_dir, '..', 'data', 'URIELPlus_Union_SoftImpute2.csv')
df = pd.read_csv(data_path)
# languages is first column
languages = df.iloc[:, 0]
# delete first column of names
X = df.iloc[:, 1:]
print("language dimensions: ", languages.shape, "\n")
print("dataframe dimensions: ", X.shape, "\n")