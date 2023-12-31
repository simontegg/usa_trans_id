import numpy as np
import pandas as pd
import statsmodels.api as sm
import re
from sklearn.impute import SimpleImputer
from utils import set_target

numerical_columns = [
        "age",
        "THHLD_NUMPER",
        "THHLD_NUMKID",
        "THHLD_NUMADLT",
        "TENROLLPUB",
        "TENROLLPRV",
        "TENROLLHMSCH",
        "TSPNDFOOD",
        "TSPNDPRPD",
        ]

ordinal_cols = [
        "EEDUC",
        "CURFOODSUF", 
        "CHILDFOOD",
        "ANXIOUS",
        "WORRY",
        "INTEREST",
        "DOWN",
        "SEEING",
        "HEARING",
        "MOBILITY",
        "REMEMBERING",
        "SELFCARE",
        "UNDERSTAND",
        "EVICT",
        "FORCLOSE",
        "ENERGY",
        "HSE_TEMP",
        "ENRGY_BILL",
        "INCOME"
        ]

def prepare_df(df, features, mf, ID, year):
    df.replace([-88, -99], np.nan, inplace=True)
    df["age"] = year - df["TBIRTH_YEAR"]
    target = f"sex-{mf}_{ID}"


    # organise features
    features = ['_'.join([part for part in re.split('(?=\d)', x)[0].split('_') if part]) for x in features]
    features = list(set(features))

    if "const" in features:
        features.remove("const")

    print(features)

    all_columns = df.columns.tolist()
    not_present = [f for f in features if f not in all_columns]

    print("not_present:")
    print(not_present)

    if len(not_present) > 0:
        return pd.DataFrame(), pd.DataFrame()


    df[target] = set_target(df, mf, ID)

    categorical_columns = [f for f in features if f not in numerical_columns and f not in ordinal_cols]

    # Subset 
    sex = df[df["EGENID_BIRTH"] == mf]
    sex_encoded = pd.DataFrame()
    
    if len(categorical_columns) > 0:
        to_encode = sex.filter(categorical_columns)
        sex_encoded = pd.get_dummies(to_encode, columns=categorical_columns, drop_first=True)

    other = sex[[f for f in features if f not in categorical_columns]]

    X = pd.concat([sex_encoded, other], axis=1)
    imputer = SimpleImputer(strategy='most_frequent')  # or 'median', 'most_frequent'
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
    X_imputed_df = X_imputed_df.astype(float)

    y = sex[target]
    X = sm.add_constant(X_imputed_df)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    return X, y
