import numpy as np
import pandas as pd
import pickle
import re
import statsmodels.api as sm
from utils import set_target
from prepare_df import prepare_df
import file_names as fn


def recursive_feature_elimination_aic(X, y):
    features = X.columns.tolist()
    best_aic = float('inf')
    best_features_aic = None

    while len(features) > 1:
        X_sm = sm.add_constant(X[features])
        model = sm.Logit(y, X_sm).fit(disp=0)
        aic = model.aic
        if aic < best_aic:
            best_aic = aic
            best_features_aic = features.copy()

        # Remove the feature with the least importance
        worst_feature = model.pvalues.drop('const').idxmax()
        print(f"removing: {worst_feature}")
        features.remove(worst_feature)

    best_features = best_features_aic.copy()

    while len(best_features) > 0:
        X_sm = sm.add_constant(X[best_features])
        model = sm.Logit(y, X_sm).fit(disp=0)  # disp=0 suppresses fit output
        # Get the max p-value
        max_pval = model.pvalues.drop('const').max()
        if max_pval < 0.05:
            break
        # Drop the feature with the highest p-value
        worst_feature = model.pvalues.drop('const').idxmax()
        print(f"removing: {worst_feature}")
        best_features.remove(worst_feature)
    
    return sm.Logit(y, sm.add_constant(X[best_features])).fit(disp=0), best_features


def run_model(sex, id, year, number, universal):
    df = pd.read_csv(f"./data/pulse{year}_puf_{number}.csv")
    feature_file = fn.features_path(sex, id, year, number, universal)
    features = pd.read_csv(feature_file)
    features = features[abs(features['Importance']) > 0.05]
    features = ['_'.join([part for part in re.split('(?=\d)', x)[0].split('_') if part]) for x in features['Feature']]
    features = list(set(features))

    X, y = prepare_df(df, features, mf, ID, year)
    model, best_features = recursive_feature_elimination_aic(X, y)
    model_summary = model.summary2().as_text()
    print(model_summary)

    model_summary_path = fn.model_summary_path(sex, id, year, number, universal)

    with open(model_summary_path, 'w') as file:
        file.write(model_summary)

    model_path = fn.model_path(sex, id, year, universal)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)












