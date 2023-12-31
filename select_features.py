import numpy as np
import pandas as pd
from pycaret.classification import *
import file_names as fn

earliest = pd.read_csv(f"./data/pulse2021_puf_34.csv")
last = pd.read_csv(f"./data/pulse2023_puf_63.csv")

set1 = set(earliest.columns)
set2 = set(last.columns)


def select_features(sex, id, year, number, universal):
    not_universal = set1.symmetric_difference(set2) if universal == True else []
    print("not_universal:")
    print(not_universal)

    df = pd.read_csv(f"./data/pulse{year}_puf_{number}.csv")
    df.replace([-88, -99], np.nan, inplace=True)

# Define the conditions
    conditions = [
        (df["EGENID_BIRTH"] == 2) & ((df["GENID_DESCRIBE"] == 1) | (df["GENID_DESCRIBE"] == 3)),  # trans_id_female
        (df["EGENID_BIRTH"] == 1) & ((df["GENID_DESCRIBE"] == 2) | (df["GENID_DESCRIBE"] == 3)),   # trans_id_male
        (df["EGENID_BIRTH"] == 2) & (df["GENID_DESCRIBE"] == 4),  # non-binary female
        (df["EGENID_BIRTH"] == 1) & (df["GENID_DESCRIBE"] == 4),  # non-binary male
    ]

# Define the choices
    choices = [
        "trans_id_female",
        "trans_id_male",
        "non_binary_female",
        "non_binary_male",
    ]

    target= "transgender_identity"
    df[target] = np.select(conditions, choices, default="no")

    unique_value_counts = df[target].value_counts()
    print("unique_value_counts:")
    print(unique_value_counts)

    df["age"] = year - df["TBIRTH_YEAR"]

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

    numerical_columns = [x for x in numerical_columns if x not in not_universal]

    ordinal_columns = [
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

    ordinal_columns = [x for x in ordinal_columns if x not in not_universal]

    ordinal = {
            "EEDUC": [1, 2, 3, 4, 5, 6, 7],
            "CURFOODSUF": [1, 2, 3, 4],
            "CHILDFOOD": [1, 2, 3],
            "ANXIOUS": [1, 2, 3, 4],
            "WORRY": [1, 2, 3, 4],
            "INTEREST": [1, 2, 3, 4],
            "DOWN": [1, 2, 3, 4],
            "SEEING": [1, 2, 3, 4],
            "HEARING": [1, 2, 3, 4],
            "MOBILITY": [1, 2, 3, 4],
            "REMEMBERING": [1, 2, 3, 4],
            "SELFCARE": [1, 2, 3, 4],
            "UNDERSTAND": [1, 2, 3, 4],
            "EVICT": [1, 2, 3, 4],
            "FORCLOSE": [1, 2, 3, 4],
            "ENERGY": [1, 2, 3, 4],
            "HSE_TEMP": [1, 2, 3, 4],
            "ENRGY_BILL": [1, 2, 3, 4],
            "INCOME": [1, 2, 3, 4, 5, 6, 7, 8]
            }

    ordinal = {key: ordinal[key] for key in ordinal if key not in not_universal}

    exclude_cols = [
            "TBIRTH_YEAR",
            "EGENID_BIRTH",
            "GENID_DESCRIBE",
            "ABIRTH_YEAR",
            "AGENID_BIRTH",
            "AHISPANIC",
            "ARACE",
            "AEDUC",
            "AHHLD_NUMPER",
            "AHHLD_NUMKID",
            "HWEIGHT",
            "PWEIGHT",
            "SCRAM",
            "WEEK"
            ]

    all_columns = df.columns.tolist()
    categorical_columns = [col for col in all_columns if col != target and (col not in numerical_columns) and (col not in exclude_cols) and (col not in ordinal_columns) and (col not in not_universal)]
    exclude = [x for x in all_columns if x in not_universal]
    exclude = exclude_cols + exclude

    print(categorical_columns)

    data = df[df["EGENID_BIRTH"] == sex]

    s = setup(
            data=data, 
            target=target, 
            ordinal_features=ordinal,
            categorical_features=categorical_columns,  
            ignore_features=exclude,
            session_id=123,
            feature_selection=True,
            )


    print(s)

# best = compare_models() #Logistic Regression
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=1000,
#                    multi_class='auto', n_jobs=None, penalty='l2',
#                    random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
#                    warm_start=False)
    model = create_model("lr")

    X_train = get_config('X_transformed')

# Extract the feature names
    selected_features = X_train.columns
    feature_importances = model.coef_[0]

    print(selected_features)
    print(feature_importances)

    features = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
    features = features.reindex(features['Importance'].abs().sort_values(ascending=False).index)
    print(features)

    features_path = fn.features_path(sex, id, year, number, universal)
    features.to_csv(features_path, float_format="%.4f")




