from datetime import date
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import dataframe_image as dfi
import re
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import statsmodels.api as sm

# Utilities
lowess = sm.nonparametric.lowess
palette = seaborn.color_palette("deep")
palette_hex = [mcolors.to_hex(color) for color in palette]


index = pd.read_csv(f"./data/index.csv")


def n(df):
    return df.shape[0]


def get_year(file):
    year_match = re.search(r'\d{4}', file)

    if year_match:
        return int(year_match.group())
    else:
        return 0


stats = []
stats_index = []
upper = 65
lower = 45

for i, row in index.iterrows():
    print(row["File"])
    file = row["File"]
    year = get_year(file)
    year_max = year - lower
    year_min = year - upper
    responses = pd.read_csv(f"./data/{file}")
    age_query = f"TBIRTH_YEAR >= {year_min} and TBIRTH_YEAR <= {year_max}"
    by_age =    responses.query(age_query)
    total =     n(by_age)
    
    males =     by_age.query("EGENID_BIRTH == 1")
    m_total =   n(males)
    m_f =       males.query("GENID_DESCRIBE == 2")
    m_trans =   males.query('GENID_DESCRIBE == 3')
    m_none =    males.query('GENID_DESCRIBE == 4')

    females =   by_age.query("EGENID_BIRTH == 2")
    f_total =   n(females)
    f_m =       females.query("GENID_DESCRIBE == 1")
    f_trans =   females.query("GENID_DESCRIBE == 3")
    f_none =    females.query("GENID_DESCRIBE == 4")

    point = {
            "date": pd.to_datetime(row["Mid point"]),
            "male_id_female": n(m_f) / m_total,
            "male_id_trans": n(m_trans) / m_total,
            "male_id_none": n(m_none) / m_total,
            "female_id_male": n(f_m) / f_total,
            "female_id_trans": n(f_trans) / f_total,
            "female_id_none": n(f_none) / f_total,
            }

    stats.append(point)

df = pd.DataFrame(stats)
df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
df = df.set_index('date')


# Analysis
# X = sm.add_constant(df['date_numeric'])  # Add a constant term to the predictor
# y = df['female_id_none']
# model = sm.OLS(y, X).fit()

# print(model.summary())




seaborn.set_theme()

# Format the y-axis labels as percentages
fmt = '{x:,.0%}'
tick_formatter = mtick.StrMethodFormatter(fmt)


print(df)

# ax = seaborn.scatterplot(x='date', data=df)
# ax = seaborn.scatterplot(x="date", y='female_id_male', data=df, label="Female ID as male")
seaborn.scatterplot(x="date", y='female_id_none', data=df, label="Female ID as non-binary")
seaborn.scatterplot(x="date", y='female_id_trans', data=df, label="Female ID as trans")
ax = seaborn.scatterplot(x="date", y='female_id_male', data=df, label="Female ID as male")
seaborn.scatterplot(x="date", y='male_id_none', data=df, label="Male ID as non-binary")
seaborn.scatterplot(x="date", y='male_id_trans', data=df, label="Male ID as trans")
seaborn.scatterplot(x="date", y='male_id_female', data=df, label="Male ID as female")

# Lowess
points = [
        "female_id_none",
        "female_id_trans",
        "female_id_male",
        "male_id_none",
        "male_id_trans",
        "male_id_female"
        ]

for i, point in enumerate(points):
    name = f"{point}_smooth"
    z = lowess(df[point], df['date_numeric'], frac=0.5)
    df[name] = pd.Series(z[:,1], index=df.index)
    seaborn.lineplot(x="date", y=name, data=df, label=None, color=palette_hex[i])



ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.MonthLocator(bymonth=(1, 7))))

# Custom formatter to display year only on the January tick
def custom_date_formatter(x, pos=None):
    date = mdates.num2date(x)
    if date.month == 1:
        return f'{date.year}'
    else:
        return ''

ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))


plt.rc('font', size=12)  
plt.ylabel("Prevalence")
plt.xlabel(None)
# plt.xlim(["2006-01-01", "2023-01-01"])
plt.xlim([pd.to_datetime("2021-01-01"), pd.to_datetime("2024-01-01")])
plt.legend(loc='upper left')






# df1.style
# dfi.export(df1, 'df_styled.png')


plt.show()




# df.to_csv(f"./results/{name}.csv", float_format="%.4f")

# # Table png
# # percent = lambda x: "{:.2f}%".format(x*100)
# # styled = df.style.format(percent)
# # dfi.export(styled, f"results/{name}.png")






