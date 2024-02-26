from enum import IntEnum

class Sex(IntEnum):
    Male = 1
    Female = 2

class Id(IntEnum):
    Trans = 1 # OppositeSex + Transgender
    NonBinary = 2
    OppositeSex = 3
    Transgender = 4

def get_id(id):
    if id == Id.Trans:
        return "trans"
    elif id == Id.NonBinary:
        return "non-binary"
    elif id == Id.OppositeSex:
        return "opp-sex"
    elif id == Id.Transgender:
        return "transgender"


def model_path(sex, id, year, number, universal):
    mf = "male" if sex == Sex.Male else "female"
    u = "universal" if universal else "survey"
    return f"./models/{mf}_{get_id(id)}_{year}_{number}_{u}.pkl"


def features_path(sex, id, year, number, universal):
    mf = "male" if sex == Sex.Male else "female"
    u = "universal" if universal else "survey"
    return f"./results/{mf}_{get_id(id)}_{year}_{number}_{u}_features.csv"


def model_summary_path(sex, id, year, number, universal):
    mf = "male" if sex == Sex.Male else "female"
    u = "universal" if universal else "survey"
    return f"./results/model_{mf}_{get_id(id)}_{year}_{number}_{u}.txt"
