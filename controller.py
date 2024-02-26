import file_names as fn
from select_features import select_features
from model import run_model


sexes = [fn.Sex.Male, fn.Sex.Female]
ids = [fn.Id.Trans, fn.Id.NonBinary]
numbers = range(34, 65)
universal_setting = [False, True]

def get_year(number):
    if number < 40:
        return 2021
    elif number < 53:
        return 2022
    elif number < 64:
        return 2023
    else:
        return 2024

for universal in universal_setting:
    for number in numbers:
        year = get_year(number)
        for sex in sexes:
            for id in ids:
                select_features(sex, id, year, number, universal)
                run_model(sex, id, year, number, universal)









