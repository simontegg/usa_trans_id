import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn
import statsmodels.api as sm
from utils import get_year


# Utilities
lowess = sm.nonparametric.lowess
palette = seaborn.color_palette("deep")
palette_hex = [mcolors.to_hex(color) for color in palette]

def n(df):
    return df.shape[0]


index = pd.read_csv(f"./data/index.csv")
stats = []
upper = 24
lower = 18
region = None # 48 = Texas

covid_symptoms = f"SYMPTMNOW == 1 | SYMPTMIMPCT == 1 | SYMPTMIMPCT == 2"
college = f"EEDUC >= 4"

for i, row in index.iterrows():
    print(row["File"])
    file = row["File"]
    year = get_year(file)
    year_max = year - lower
    year_min = year - upper
    responses = pd.read_csv(f"./data/{file}")
    age_query = f"TBIRTH_YEAR >= {year_min} and TBIRTH_YEAR <= {year_max}"

    if region is not None:
        region_query = f"EST_ST == {region}"
        responses = responses.query(region_query)

    by_age =    responses.query(age_query)

    total =     n(by_age)
    
    males =     by_age.query("EGENID_BIRTH == 1")
    m_total =   n(males)
    m_f =       males.query("GENID_DESCRIBE == 2")
    m_trans =   males.query('GENID_DESCRIBE == 3')
    m_f_trans = males.query("GENID_DESCRIBE == 2 | GENID_DESCRIBE == 3")
    m_none =    males.query('GENID_DESCRIBE == 4')

    females =   by_age.query("EGENID_BIRTH == 2")
    f_total =   n(females)
    f_m =       females.query("GENID_DESCRIBE == 1")
    f_trans =   females.query("GENID_DESCRIBE == 3")
    f_m_trans = females.query("GENID_DESCRIBE == 1 | GENID_DESCRIBE == 3")
    f_none =    females.query("GENID_DESCRIBE == 4")

    females_white = females.query("RRACE == 1")
    f_m_trans_white = f_m_trans.query("RRACE == 1")
    f_none_white = f_none.query("RRACE == 1")
    males_white = males.query("RRACE == 1")
    m_f_trans_white = m_f_trans.query("RRACE == 1")
    m_none_white = m_none.query("RRACE == 1")

    females_college = females.query(college)
    f_m_trans_college = f_m_trans.query(college)
    males_college = males.query(college)
    m_f_trans_college = m_f_trans.query(college)

    male_lg = males.query("SEXUAL_ORIENTATION == 1")
    female_lg = females.query("SEXUAL_ORIENTATION == 1")
    male_lg_trans = male_lg.query("GENID_DESCRIBE == 2 | GENID_DESCRIBE == 3")
    female_lg_trans = male_lg.query("GENID_DESCRIBE == 1 | GENID_DESCRIBE == 3")

    point = {
            "date": pd.to_datetime(row["Mid point"]),
            "male_id_female": n(m_f) / m_total,
            "male_id_trans": n(m_trans) / m_total,
            "male_id_none": n(m_none) / m_total,
            "female_id_male": n(f_m) / f_total,
            "female_id_trans": n(f_trans) / f_total,
            "female_id_none": n(f_none) / f_total,
            "female_white": n(females_white) / f_total,
            "female_trans_white": n(f_m_trans_white) / n(f_m_trans),
            "female_none_white": n(f_none_white) / n(f_none),
            "female_college": n(females_college) / f_total,
            "female_trans_college": n(f_m_trans_college) / n(f_m_trans),
            "male_college": n(males_college) / m_total,
            "male_trans_college": n(m_f_trans_college) / n(m_f_trans),
            "male_white": n(males_white) / m_total,
            "male_trans_white": n(m_f_trans_white) / n(m_f_trans),
            "male_none_white": n(m_none_white) / n(m_none),
            }

    if "SYMPTMNOW" in responses.columns and "SYMPTMIMPCT" in responses.columns:
        males_c19 = males.query(covid_symptoms)
        m_f_trans_c19 =   m_f_trans.query(covid_symptoms)
        females_c19 = females.query(covid_symptoms)
        f_m_trans_c19 =   f_m_trans.query(covid_symptoms)
        m_none_c19 = m_none.query(covid_symptoms)
        f_none_c19 = f_none.query(covid_symptoms)
        m_homo_c19 = male_lg.query(covid_symptoms)
        f_homo_c19 = female_lg.query(covid_symptoms)

        point["male_with_symptoms"] = n(males_c19) / m_total
        point["male_trans_with_symptoms"] = n(m_f_trans_c19) / n(m_f_trans)
        point["male_none_with_symptoms"] = n(m_none_c19) / n(m_none)
        point["female_with_symptoms"] = n(females_c19) / f_total
        point["female_trans_with_symptoms"] = n(f_m_trans_c19) / n(f_m_trans)
        point["female_none_with_symptoms"] = n(f_none_c19) / n(f_none)
        point["female_homo_with_symptoms"] = n(f_homo_c19) / n(female_lg)
        point["male_homo_with_symptoms"] = n(m_homo_c19) / n(male_lg)
        

    stats.append(point)

df = pd.DataFrame(stats)
df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
df = df.set_index('date')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)


# Analysis
# X = sm.add_constant(df['date_numeric'])  # Add a constant term to the predictor
# y = df['female_id_none']
# model = sm.OLS(y, X).fit()

# print(model.summary())


# Chart
seaborn.set_theme()

ax = seaborn.scatterplot(x="date", y='female_id_none', data=df, label="Female ID as non-binary")
seaborn.scatterplot(x="date", y='female_id_trans', data=df, label="Female ID as trans")
seaborn.scatterplot(x="date", y='female_id_male', data=df, label="Female ID as male")
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


fmt = '{x:,.0%}'
tick_formatter = mtick.StrMethodFormatter(fmt)

def custom_date_formatter(x, pos=None):
    date = mdates.num2date(x)
    if date.month == 1:
        return f'{date.year}'
    else:
        return ''


ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
ax.yaxis.set_major_formatter(tick_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
ax.set_title(f"Trangender and non-binary identification in {lower} to {upper} year olds")

plt.rc('font', size=12)  
plt.ylabel("Prevalence")
plt.xlabel(None)
plt.ylim([0, 0.09])
plt.xlim([pd.to_datetime("2021-01-01"), pd.to_datetime("2024-01-01")])
plt.legend(loc='upper left')

plt.show()


# Write results
name = f"{lower}_{upper}_trans_id"
df.to_csv(f"./results/{name}.csv", float_format="%.4f")







