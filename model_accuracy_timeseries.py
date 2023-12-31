import numpy as np
import pandas as pd
import pickle
import re
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from utils import set_target
from prepare_df import prepare_df

mf = 2
ID = "non_binary"
year = 2021
number = 34
universal = False


with open(f"./models/sex-{mf}_{ID}_{year}_{number}.pkl", 'rb') as file:
    model = pickle.load(file)
    features = model.params.index.tolist()
    all_features = [feature for feature in features if feature != 'const']
    index = pd.read_csv(f"./data/index.csv")

    stats = []

    for i, row in index.iterrows():
        print(row["File"])
        file = row["File"]
        df = pd.read_csv(f"./data/{file}")
        X, y = prepare_df(df, features, mf, ID, year)

        if X.empty:
            continue

        target = f"sex-{mf}_{ID}"

        y_pred = model.predict(X[features])
        y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

        accuracy = accuracy_score(y, y_pred_binary)
        print(f"Model Accuracy: {accuracy}")

        file_parts = file.split("_")
        file_num = int(file_parts[-1].replace(".csv", ""))

        stats_row = {
                "accuracy": accuracy,
                "date": row["Mid point"],
                "origin": file_num == number
                }

        stats.append(stats_row)


    stats_df = pd.DataFrame(stats)
    print(stats_df)
    stats_df.to_csv(f"./results/model_accuracy_{mf}_{ID}.csv", float_format="%.4f")











