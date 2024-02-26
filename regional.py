
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import ruptures as rpt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import seaborn
import statsmodels.api as sm
from utils import get_year
import math


# Utilities
lowess = sm.nonparametric.lowess
palette = seaborn.color_palette("deep")
palette_hex = [mcolors.to_hex(color) for color in palette]

lt_18_prohibitions_2022_2023 = [
        1,
        19,
        21,
        22,
        28,
        29,
        37,
        38,
        40,
        46,
        47,
        48,
        54
        ]

no_probitions = [
        2,
        4,
        6,
        8,
        9,
        10,
        11,
        13,
        15,
        17,
        20,
        23,
        24,
        25,
        26,
        27,
        32,
        33,
        34,
        35,
        36,
        41,
        42,
        44,
        45,
        50,
        51,
        53,
        55,
        56
        ]

regions = [
        # lt_18_prohibitions_2022_2023, 
        # no_probitions,
        None,

        ]


def n(df):
    return df.shape[0]

index = pd.read_csv(f"./data/index.csv")
stats = []
investigation = []
upper = 24
lower = 18

covid_symptoms = f"SYMPTMNOW == 1 | SYMPTMIMPCT == 1 | SYMPTMIMPCT == 2"
college = f"EEDUC >= 4"

for region in regions:
    for i, row in index.iterrows():
        print(row["File"])
        file = row["File"]
        year = get_year(file)
        year_max = year - lower
        year_min = year - upper
        responses = pd.read_csv(f"./data/{file}")
        age_query = f"TBIRTH_YEAR >= {year_min} and TBIRTH_YEAR <= {year_max}"

        by_region = responses if region is None else responses[responses['EST_ST'].isin(region)]
        by_age =    by_region.query(age_query)

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

        point = {
                "date": pd.to_datetime(row["Mid point"]),
                "male_id_female": n(m_f) / m_total,
                "male_id_trans": n(m_trans) / m_total,
                "male_id_none": n(m_none) / m_total,
                "female_id_male": n(f_m) / f_total,
                "female_id_trans": n(f_trans) / f_total,
                "female_id_none": n(f_none) / f_total,
                }
        point["male_id_tf"] = point["male_id_trans"] + point["male_id_female"]
        point["female_id_tm"] = point["female_id_trans"] + point["female_id_male"]

        stats.append(point)

df = pd.DataFrame(stats)
df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
df = df.set_index('date')



df_reset = df.reset_index()
series = [
        # "male_id_female",
        # "male_id_trans",
        "male_id_tf",
        # "male_id_none",
        # "female_id_male",
        # "female_id_trans",
        # "female_id_tm",
        # "female_id_none"
        ]

for s in series:
    values = df[s].values

    algo = rpt.Pelt(model="rbf").fit(values)
    result = algo.predict(pen=10)
    print(f"Breakpoints '{s}' at indices:", result)


    prophet_subset = df_reset[['date', s]].rename(columns={'date': 'ds', s: 'y'})
    prophet_subset["cap"] = 0.2
    prophet_subset["floor"] = 0
    model = Prophet(
            growth='logistic',
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=0.05,
            changepoint_range=1,
            daily_seasonality=False, 
            weekly_seasonality=False
            # changepoints=["2023-04-04"]
            )
    model.fit(prophet_subset)
    future = model.make_future_dataframe(periods=0)
    future["cap"] = 0.2
    future["floor"] = 0
    forecast = model.predict(future)
    max_yhat = forecast["yhat"].max()
    max_y = prophet_subset["y"].max()
    # fig = model.plot_components(forecast)
    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_ylim([0, math.ceil((max(max_y, max_yhat) + 0.01) * 100) / 100])
    a = add_changepoints_to_plot(ax, model, forecast)
    plt.show()






# Optional: Plotting
# rpt.display(series, result)
# plt.show()







# Analysis
# X = sm.add_constant(df['date_numeric'])  # Add a constant term to the predictor
# y = df['female_id_none']
# model = sm.OLS(y, X).fit()

# print(model.summary())


# Chart
# seaborn.set_theme()

# ax = seaborn.scatterplot(x="date", y='female_id_none', data=df, label="Female ID as non-binary")
# seaborn.scatterplot(x="date", y='female_id_trans', data=df, label="Female ID as trans")
# seaborn.scatterplot(x="date", y='female_id_male', data=df, label="Female ID as male")
# seaborn.scatterplot(x="date", y='male_id_none', data=df, label="Male ID as non-binary")
# seaborn.scatterplot(x="date", y='male_id_trans', data=df, label="Male ID as trans")
# seaborn.scatterplot(x="date", y='male_id_female', data=df, label="Male ID as female")

# Lowess
# points = [
#         "female_id_none",
#         "female_id_trans",
#         "female_id_male",
#         "male_id_none",
#         "male_id_trans",
#         "male_id_female"
#         ]

# for i, point in enumerate(points):
#     name = f"{point}_smooth"
#     z = lowess(df[point], df['date_numeric'], frac=0.5)
#     df[name] = pd.Series(z[:,1], index=df.index)
#     seaborn.lineplot(x="date", y=name, data=df, label=None, color=palette_hex[i])


# fmt = '{x:,.0%}'
# tick_formatter = mtick.StrMethodFormatter(fmt)

# def custom_date_formatter(x, pos=None):
#     date = mdates.num2date(x)
#     if date.month == 1:
#         return f'{date.year}'
#     else:
#         return ''


# ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
# ax.yaxis.set_major_formatter(tick_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
# ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
# ax.set_title(f"Trangender and non-binary identification in {lower} to {upper} year olds")

# plt.rc('font', size=12)  
# plt.ylabel("Prevalence")
# plt.xlabel(None)
# plt.ylim([0, 0.09])
# plt.xlim([pd.to_datetime("2021-01-01"), pd.to_datetime("2024-03-01")])
# plt.legend(loc='upper left')

# plt.show()


# Write results
name = f"{lower}_{upper}_trans_id"
df.to_csv(f"./results/{name}.csv", float_format="%.4f")



        # 1,
        # 19,
        # 21,
        # 22,
        # 28,
        # 29,
        # 30,
        # 37,
        # 38,
        # 40,
        # 46,
        # 47,
        # 48,
        # 54




                    # '01'='Alabama'
                    # '02'='Alaska'
                    # '04'='Arizona'
                    # '05'='Arkansas'
                    # '06'='California'
                    # '08'='Colorado'
                    # '09'='Connecticut'
                    # '10'='Delaware'
                    # '11'='District of Columbia'
                    # '12'='Florida'
                    # '13'='Georgia'
                    # '15'='Hawaii'
                    # '16'='Idaho'
                    # '17'='Illinois'
                    # '18'='Indiana'
                    # '19'='Iowa'
                    # '20'='Kansas'
                    # '21'='Kentucky'
                    # '22'='Louisiana'
                    # '23'='Maine'
                    # '24'='Maryland'
                    # '25'='Massachusetts'
                    # '26'='Michigan'
                    # '27'='Minnesota'
                    # '28'='Mississippi'
                    # '29'='Missouri'
                    # '30'='Montana'
                    # '31'='Nebraska'
                    # '32'='Nevada'
                    # '33'='New Hampshire'
                    # '34'='New Jersey'
                    # '35'='New Mexico'
                    # '36'='New York'
                    # '37'='North Carolina'
                    # '38'='North Dakota'
                    # '39'='Ohio'
                    # '40'='Oklahoma'
                    # '41'='Oregon'
                    # '42'='Pennsylvania'
                    # '44'='Rhode Island'
                    # '45'='South Carolina'
                    # '46'='South Dakota'
                    # '47'='Tennessee'
                    # '48'='Texas'
                    # '49'='Utah'
                    # '50'='Vermont'
                    # '51'='Virginia'
                    # '53'='Washington'
                    # '54'='West Virginia'
                    # '55'='Wisconsin'
                    # '56'='Wyoming'

