"""Calendar, COVID-era and weather features shared by training and inference."""
import numpy as np
import pandas as pd
import holidays

NSW_HOL = holidays.Australia(subdiv="NSW", years=range(2012, 2031))
LOCKDOWNS = [("2020-03-23", "2020-05-15"), ("2021-06-26", "2021-10-11")]
RESTRICTIONS = ("2020-03-16", "2022-02-28")  # public-health orders / WFH guidance era
DAYTYPES = ["mon_thu", "fri", "sat", "sun", "holiday"]
NAMES = ([f"h_sin{k}" for k in range(1, 5)] + [f"h_cos{k}" for k in range(1, 5)] +
         [f"dt_{d}" for d in DAYTYPES] + ["m_sin", "m_cos", "m_sin2", "m_cos2", "year",
         "lockdown", "restrictions", "temp", "rain", "cloud", "wind"])

def daytype_matrix(dates):
    """dates: DatetimeIndex (normalised days). Returns (n,5) one-hot."""
    dow = dates.dayofweek.values
    m = np.zeros((len(dates), 5), np.float32)
    hol = np.array([d in NSW_HOL for d in dates.date])
    m[(dow <= 3) & ~hol, 0] = 1
    m[(dow == 4) & ~hol, 1] = 1
    m[(dow == 5) & ~hol, 2] = 1
    m[(dow == 6) & ~hol, 3] = 1
    m[hol, 4] = 1
    return m

def covid_flags(dates):
    d = pd.DatetimeIndex(dates)
    lock = np.zeros(len(d), np.float32)
    for a, b in LOCKDOWNS:
        lock[(d >= a) & (d <= b)] = 1
    restr = ((d >= RESTRICTIONS[0]) & (d <= RESTRICTIONS[1])).astype(np.float32)
    return lock, restr

def build(hour, daytype, month, year_frac, lockdown, restrictions, temp, rain, cloud, wind):
    hour = np.asarray(hour, np.float32); month = np.asarray(month, np.float32)
    cols = [np.sin(2 * np.pi * k * hour / 24) for k in range(1, 5)] + \
           [np.cos(2 * np.pi * k * hour / 24) for k in range(1, 5)]
    cols += list(np.asarray(daytype, np.float32).T)
    cols += [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
             np.sin(4 * np.pi * month / 12), np.cos(4 * np.pi * month / 12)]
    cols += [(np.asarray(year_frac, np.float32) - 2020) / 5, lockdown, restrictions,
             np.asarray(temp, np.float32) / 30, np.log1p(np.asarray(rain, np.float32)),
             np.asarray(cloud, np.float32) / 100, np.asarray(wind, np.float32) / 50]
    return np.column_stack(cols).astype(np.float32)
