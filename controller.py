import file_names as fn
from select_features import select_features


sexes = [fn.Sex.Male, fn.Sex.Female]
ids = [fn.Id.Trans, fn.Id.NonBinary]
numbers = range(34, 64)
universal_setting = [False, True]

def get_year(number):
    if number < 40:
        return 2021
    elif number < 53:
        return 2022
    else:
        return 2023

for universal in universal_setting:
    for number in numbers:
        year = get_year(number)
        for sex in sexes:
            for id in ids:
                select_features(sex, id, year, number, universal)








