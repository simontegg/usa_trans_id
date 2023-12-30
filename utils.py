import re
import numpy as np

def set_target(df, sex, ID):
    conditions = [
        (df["EGENID_BIRTH"] == 2) & ((df["GENID_DESCRIBE"] == 1) | (df["GENID_DESCRIBE"] == 3)),  # trans_id_female
        (df["EGENID_BIRTH"] == 1) & ((df["GENID_DESCRIBE"] == 2) | (df["GENID_DESCRIBE"] == 3)),   # trans_id_male
        (df["EGENID_BIRTH"] == 2) & (df["GENID_DESCRIBE"] == 4),  # non-binary female
        (df["EGENID_BIRTH"] == 1) & (df["GENID_DESCRIBE"] == 4),  # non-binary male
    ]
    choices = [0, 0, 0, 0]

    if sex == 2 and ID == "trans":
        choices[0] = 1
    elif sex == 1 and ID == "trans":
        choices[1] = 1
    elif sex == 2 and ID == "non_binary":
        choices[2] = 1
    elif sex == 1 and ID == "non_binary":
        choices[3] = 1
    else:
        raise Error("sex or id have been misspecified") 

    return np.select(conditions, choices, default=0)


def get_year(file):
    year_match = re.search(r'\d{4}', file)

    if year_match:
        return int(year_match.group())
    else:
        return 0





