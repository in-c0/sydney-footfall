"""Download every raw dataset used by the model into data/raw/.

Sources
  City of Sydney  - Walking count surveys (biannual, ~125 sites, 2013-)      ArcGIS FeatureServer
  City of Sydney  - Walking count sites (point geometry)                       ArcGIS FeatureServer
  City of Sydney  - Automatic hourly pedestrian counts (4 sensors, 2020-2025)  ArcGIS FeatureServer
  Liverpool City Council - CCTV camera people counters (hourly, 2023-)         Opendatasoft API
  Open-Meteo      - Hourly historical weather (ERA5-based archive)             archive-api.open-meteo.com
Run fetch_osm.py separately for OpenStreetMap context.
"""
import csv, json, urllib.parse, urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
ARC = "https://services1.arcgis.com/cNVyNtjGVZybOQWZ/arcgis/rest/services"
LIV = ("https://liverpool-city-council-westernparklands.opendatasoft.com/api/explore/v2.1/"
       "catalog/datasets/ncounter/exports/csv")
WEATHER_POINTS = {"sydney_cbd": (-33.870, 151.207), "liverpool": (-33.923, 150.924)}

def get(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "sydney-pedestrian-model/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read()

def arcgis(svc, out, geo=False, oid="OBJECTID"):
    rows, off = [], 0
    while True:
        q = dict(where="1=1", outFields="*", f="json", resultOffset=off,
                 resultRecordCount=2000, orderByFields=oid)
        if geo:
            q.update(returnGeometry="true", outSR="4326")
        d = json.loads(get(f"{ARC}/{svc}/FeatureServer/0/query?" + urllib.parse.urlencode(q)))
        feats = d.get("features", [])
        for f in feats:
            a = f["attributes"]
            if geo and f.get("geometry"):
                a["lon"], a["lat"] = f["geometry"]["x"], f["geometry"]["y"]
            rows.append(a)
        off += len(feats)
        if not feats or not d.get("exceededTransferLimit"):
            break
    with open(RAW / out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{out}: {len(rows)} rows")

def fetch_weather():
    for name, (lat, lon) in WEATHER_POINTS.items():
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": "2013-01-01",
            "end_date": "2026-09-30", "timezone": "Australia/Sydney",
            "hourly": "temperature_2m,precipitation,cloud_cover,wind_speed_10m"})
        (RAW / f"weather_{name}.json").write_bytes(get("https://archive-api.open-meteo.com/v1/archive?" + q))
        print(f"weather_{name}.json written")

def main():
    arcgis("WalkingCountSites", "cos_walking_sites.csv", geo=True)
    arcgis("Automatic_Walking_Count_Sites", "cos_auto_sites.csv", geo=True)
    arcgis("Walking_count_surveys", "cos_walking_surveys.csv")
    arcgis("Automatic_Hourly_Pedestrian_Count", "cos_auto_hourly.csv", oid="ObjectId")
    q = urllib.parse.urlencode({"delimiter": ",", "where": "device_name is not null",
        "select": "device_name,dev_id,device_type,datetime,new,total,current,geolocation,skip_record_flag"})
    (RAW / "liverpool_ncounter.csv").write_bytes(get(f"{LIV}?{q}", timeout=900))
    print("liverpool_ncounter.csv written")
    fetch_weather()

if __name__ == "__main__":
    main()
