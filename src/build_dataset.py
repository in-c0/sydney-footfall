"""Harmonise all raw pedestrian count sources into one hourly training table.

Output: data/processed/obs.parquet (one row per site-hour) and data/processed/sites.parquet
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import SpatialFeaturizer
import temporal as T

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = ["cos_survey", "cos_sensor", "liverpool_camera"]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"], 1)}

def weather(name):
    d = json.loads((RAW / f"weather_{name}.json").read_text())["hourly"]
    w = pd.DataFrame(d); w["time"] = pd.to_datetime(w["time"])
    return w.rename(columns={"temperature_2m": "temp", "precipitation": "rain",
                             "cloud_cover": "cloud", "wind_speed_10m": "wind"}).set_index("time")

def drop_outage_days(df, key):
    """Remove site-days whose total is < 5% of that site's median daily total (sensor down)."""
    daily = df.groupby([key, df.ts.dt.normalize()])["y"].transform("sum")
    med = df.assign(_d=daily).groupby(key)["_d"].transform("median")
    hours = df.groupby([key, df.ts.dt.normalize()])["y"].transform("size")
    return df[(daily >= 0.05 * med) & (hours >= 20)]

def sensor_rows(df, wx, source):
    df = df.copy()
    df["source"] = source
    df["hour"] = df.ts.dt.hour
    day = df.ts.dt.normalize()
    dt = T.daytype_matrix(pd.DatetimeIndex(day))
    lock, restr = T.covid_flags(day)
    w = wx.reindex(df.ts.dt.floor("h")).ffill().bfill()
    yf = df.ts.dt.year + (df.ts.dt.dayofyear - 1) / 365.25
    df[T.NAMES] = T.build(df.ts.dt.hour, dt, df.ts.dt.month, yf, lock, restr,
                          w.temp.values, w.rain.values, w.cloud.values, w.wind.values)
    return df

def main():
    wx_syd, wx_liv = weather("sydney_cbd"), weather("liverpool")
    sites = []

    # --- City of Sydney biannual surveys (10-min samples x6 -> hourly) ---
    s = pd.read_csv(RAW / "cos_walking_surveys.csv")
    geo = pd.read_csv(RAW / "cos_walking_sites.csv").rename(columns={"Site_ID": "SiteID"})
    s = s.merge(geo[["SiteID", "lat", "lon"]], on="SiteID")
    s["month"] = s.DateFilter.str.split(" - ").str[-1].map(MONTHS)
    hcols = [c for c in s.columns if c.startswith("Time_")]
    s = s.melt(id_vars=["SiteID", "Location", "Description", "Period", "Year", "month", "lat", "lon"],
               value_vars=hcols, var_name="h", value_name="y").dropna(subset=["y"])
    s["hour"] = s.h.str[5:7].astype(int)
    s["site_key"] = "cos_survey_" + s.SiteID.astype(str)
    s["source"] = "cos_survey"
    # Weekday surveys are mid-week; weekend day is not published, so split evenly Sat/Sun.
    dt = np.where((s.Period == "weekday").values[:, None], [[1, 0, 0, 0, 0]], [[0, 0, .5, .5, 0]])
    # Surveys run in fair weather: use dry-hour climatology for that month & hour.
    dry = wx_syd[wx_syd.rain == 0]
    clim = dry.groupby([dry.index.month, dry.index.hour])[["temp", "cloud", "wind"]].median()
    c = clim.reindex(pd.MultiIndex.from_arrays([s.month, s.hour])).values
    yr = s.Year + (s.month - 0.5) / 12
    lock, restr = T.covid_flags(pd.to_datetime(dict(year=s.Year, month=s.month, day=15)))
    s[T.NAMES] = T.build(s.hour, dt, s.month, yr, lock, restr, c[:, 0], np.zeros(len(s)), c[:, 1], c[:, 2])
    s["ts"] = pd.NaT
    sites.append(s.groupby("site_key").agg(name=("Location", "first"), desc=("Description", "first"),
                 lat=("lat", "first"), lon=("lon", "first")).assign(source="cos_survey"))

    # --- City of Sydney automatic sensors (hourly, 2020-2025) ---
    a = pd.read_csv(RAW / "cos_auto_hourly.csv")
    a["ts"] = pd.to_datetime(a.Date, unit="ms")  # stored as local wall-clock time
    a = a.rename(columns={"TotalCount": "y"}).drop_duplicates(["Location_code", "ts"])
    a = a[~((a.Location_code == "A004") & (a.ts >= "2022-07-03") & (a.ts < "2022-11-12"))]  # known half-count fault
    a["site_key"] = "cos_sensor_" + a.Location_code
    a = drop_outage_days(a, "site_key")
    ag = pd.read_csv(RAW / "cos_auto_sites.csv")
    a = a.merge(ag[["SiteID", "lat", "lon"]], left_on="Location_code", right_on="SiteID")
    a = sensor_rows(a, wx_syd, "cos_sensor")
    sites.append(a.groupby("site_key").agg(name=("Location_Name", "first"), lat=("lat", "first"),
                 lon=("lon", "first")).assign(desc="Automatic counter", source="cos_sensor"))

    # --- Liverpool CBD CCTV people counters (hourly, 2023-) ---
    l = pd.read_csv(RAW / "liverpool_ncounter.csv")
    l = l[(l.device_type == "Camera Counter") & l.new.notna()]
    l["ts"] = pd.to_datetime(l.datetime, utc=True).dt.tz_convert("Australia/Sydney").dt.tz_localize(None)
    l[["lat", "lon"]] = l.geolocation.str.split(",", expand=True).astype(float)
    l = l.rename(columns={"new": "y"}).drop_duplicates(["dev_id", "ts"])
    l["site_key"] = "liverpool_" + l.dev_id
    l = drop_outage_days(l, "site_key")
    l = sensor_rows(l, wx_liv, "liverpool_camera")
    sites.append(l.groupby("site_key").agg(name=("device_name", "first"), lat=("lat", "first"),
                 lon=("lon", "first")).assign(desc="CCTV people counter", source="liverpool_camera"))

    keep = ["site_key", "source", "ts", "hour", "y"] + T.NAMES
    obs = pd.concat([s[keep], a[keep], l[keep]], ignore_index=True)
    obs = obs[obs.y >= 0]
    sites = pd.concat(sites).reset_index()
    F = SpatialFeaturizer()
    sites[F.names] = F(sites.lat.values, sites.lon.values)
    sites.to_parquet(OUT / "sites.parquet", index=False)
    obs.to_parquet(OUT / "obs.parquet", index=False)
    print(obs.groupby("source").agg(rows=("y", "size"), sites=("site_key", "nunique"),
                                    mean=("y", "mean"), max=("y", "max")))

if __name__ == "__main__":
    main()
