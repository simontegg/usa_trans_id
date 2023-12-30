import numpy as np
import pandas as pd
import pickle
import re
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from utils import set_target

mf = 2
ID = "trans"
year = 2023
number = 58


with open(f"./models/sex-{mf}_{ID}_{year}_{number}.pkl", 'fb') as file:
    model = pickle.load(file)
    index = pd.read_csv(f"./data/index.csv")
    for i, row in index.iterrows():
        print(row["File"])
        file = row["File"]
        df = pd.read_csv(f"./data/{file}")
        df.replace([-88, -99], np.nan, inplace=True)
        df["age"] = year - df["TBIRTH_YEAR"]
        target= f"sex-{mf}_{ID}"
        df[target] = set_target(df, mf, ID)







