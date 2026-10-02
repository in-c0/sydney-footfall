"""Inference: predict hourly footfall anywhere in the study areas.

  python predict.py --lat -33.8737 --lon 151.2069 --daytype sat
  python predict.py --export-web   # street-segment predictions for the web demo
"""
import argparse, base64, json, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from features import SpatialFeaturizer, to_xy, ROAD_CLASS
from model import FootfallNet, SOURCES
import temporal as T

ROOT = Path(__file__).resolve().parents[1]
MODELS, RAW, PROC = ROOT / "models", ROOT / "data" / "raw", ROOT / "data" / "processed"
DT_VEC = {"mon_thu": [1, 0, 0, 0, 0], "fri": [0, 1, 0, 0, 0], "sat": [0, 0, 1, 0, 0],
          "sun": [0, 0, 0, 1, 0], "holiday": [0, 0, 0, 0, 1]}

class Ensemble:
    def __init__(self):
        self.members = []
        for p in sorted(MODELS.glob("footfall_seed*.pt")):
            ck = torch.load(p, weights_only=False)
            net = FootfallNet(len(ck["spatial_names"]), len(ck["temporal_names"]))
            net.load_state_dict(ck["state_dict"]); net.eval()
            self.members.append((net, ck))
        self.feat = SpatialFeaturizer()

    @torch.no_grad()
    def __call__(self, XS, XT, source="cos_survey"):
        src = torch.full((len(XS),), SOURCES.index(source), dtype=torch.long)
        outs = []
        for net, ck in self.members:
            xs = torch.from_numpy(((XS - ck["spatial_mean"]) / ck["spatial_std"]).astype(np.float32))
            xt = torch.from_numpy(((XT - ck["temporal_mean"]) / ck["temporal_std"]).astype(np.float32))
            outs.append(np.concatenate([net(xs[i:i+65536], xt[i:i+65536], src[i:i+65536]).numpy()
                                        for i in range(0, len(xs), 65536)]))
        return np.expm1(np.clip(np.mean(outs, 0), 0, 12))   # people per hour

def scenario(daytype="mon_thu", month=10, year=2026, fair=True, rain=0.0):
    """Temporal features for 24 hours of a typical day."""
    w = json.loads((RAW / "weather_sydney_cbd.json").read_text())["hourly"]
    w = pd.DataFrame(w); w["time"] = pd.to_datetime(w["time"])
    w = w[(w.time.dt.month == month) & ((w.precipitation == 0) if fair else True)]
    c = w.groupby(w.time.dt.hour)[["temperature_2m", "cloud_cover", "wind_speed_10m"]].median().values
    h = np.arange(24)
    return T.build(h, np.tile(DT_VEC[daytype], (24, 1)), np.full(24, month), np.full(24, year + (month - .5) / 12),
                   np.zeros(24), np.zeros(24), c[:, 0], np.full(24, rain), c[:, 1], c[:, 2])

def predict_points(ens, lat, lon, daytype="mon_thu", **kw):
    XS = ens.feat(np.atleast_1d(lat), np.atleast_1d(lon))
    XT = scenario(daytype, **kw)
    n = len(XS)
    p = ens(np.repeat(XS, 24, 0), np.tile(XT, (n, 1)))
    return p.reshape(n, 24)

def street_segments(area, bbox, max_len=60):
    """Split OSM walkable ways into <= max_len metre segments; return geometry + midpoints."""
    s, w, n, e = bbox
    segs, mids, cls = [], [], []
    keep = {"ped", "major", "minor"}
    for el in json.loads((RAW / "osm" / f"{area}_roads.json").read_text()):
        rc = ROAD_CLASS.get(el.get("tags", {}).get("highway"))
        g = el.get("geometry")
        if rc not in keep or not g: continue
        lat = np.array([p["lat"] for p in g]); lon = np.array([p["lon"] for p in g])
        if lat.max() < s or lat.min() > n or lon.max() < w or lon.min() > e: continue
        xy = to_xy(lat, lon)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        k = max(1, int(np.ceil(d[-1] / max_len)))
        cuts = np.linspace(0, d[-1], k + 1)
        for a, b in zip(cuts[:-1], cuts[1:]):
            inner = (d > a) & (d < b)
            la = np.r_[np.interp(a, d, lat), lat[inner], np.interp(b, d, lat)]
            lo = np.r_[np.interp(a, d, lon), lon[inner], np.interp(b, d, lon)]
            m = (a + b) / 2
            mlat, mlon = np.interp(m, d, lat), np.interp(m, d, lon)
            if not (s <= mlat <= n and w <= mlon <= e): continue
            segs.append(np.c_[la, lo]); mids.append((mlat, mlon)); cls.append(rc)
    return segs, np.array(mids), cls

def export_web(out=ROOT / "web" / "data.json"):
    ens = Ensemble()
    sites = pd.read_parquet(PROC / "sites.parquet")
    areas = {"sydney_cbd": (-33.925, 151.170, -33.855, 151.232),
             "liverpool": (-33.930, 150.915, -33.915, 150.932)}
    data = {"daytypes": ["mon_thu", "fri", "sat", "sun"], "scale": 25,
            "note": ("binary arrays are base64(zlib). pred: int8 hour-deltas of q=round(25*log1p(people/hour)), "
                     "shape (segments,24); pts: int16 lat/lon in 1e-5 deg, first point vs origin then deltas. "
                     "Scenario: October 2026, fair weather, two-way footpath count."),
            "areas": {}}
    for area, bbox in areas.items():
        segs, mids, cls = street_segments(area, bbox)
        origin = np.array([bbox[0], bbox[1]])
        # geometry: int16 (lat,lon) pairs in 1e-5 deg from origin, first point absolute then deltas
        npts, flat = [], []
        for sgm in segs:
            q = np.round((sgm - origin) * 1e5).astype(np.int64)
            q = np.r_[q[:1], np.diff(q, axis=0)]
            npts.append(len(q)); flat.append(q)
        flat = np.clip(np.vstack(flat), -32768, 32767).astype("<i2")
        b64 = lambda a: base64.b64encode(zlib.compress(a.tobytes(), 9)).decode()  # zlib ('deflate')
        preds = {}
        for dt in data["daytypes"]:
            p = predict_points(ens, mids[:, 0], mids[:, 1], dt)
            q = np.clip(np.round(25 * np.log1p(p)), 0, 255).astype(np.int16)
            preds[dt] = b64(np.diff(q, axis=1, prepend=0).astype(np.int8))  # hour-deltas, (seg, 24)
        data["areas"][area] = {"bbox": bbox, "origin": origin.tolist(),
                               "npts": b64(np.array(npts, np.uint8)), "pts": b64(flat),
                               "cls": "".join({"ped": "p", "major": "M", "minor": "m"}[c] for c in cls),
                               "pred": preds}
        print(area, len(segs), "segments")
    # Observed survey/sensor profiles + out-of-fold (unseen-site) predictions for validation markers
    obs = pd.read_parquet(PROC / "obs.parquet").reset_index(drop=True)
    cv = np.load(MODELS / "cv_pred_net.npy")
    obs["cv"] = np.expm1(np.clip(cv, 0, 12))
    obs["dt"] = np.where(obs.dt_mon_thu + obs.dt_fri > 0.5, "weekday", "weekend")
    out_sites = []
    for _, s in sites.iterrows():
        o = obs[obs.site_key == s.site_key]
        if s.source == "cos_survey":
            o = o[o.year == o.year.max()]
        else:
            o = o[o.ts >= o.ts.max() - pd.Timedelta(days=28)]
        prof = {}
        for dt, g in o.groupby("dt"):
            gg = g.groupby("hour")[["y", "cv"]].mean().reindex(range(24))
            prof[dt] = {"obs": [None if np.isnan(v) else int(v) for v in gg.y],
                        "cv": [None if np.isnan(v) else int(v) for v in gg.cv]}
        yr = int(round(2020 + 5 * o.year.max())) if len(o) else None
        out_sites.append({"id": s.site_key, "name": s["name"], "desc": s.desc, "source": s.source,
                          "lat": round(s.lat, 6), "lon": round(s.lon, 6), "year": yr, "profile": prof})
    data["sites"] = out_sites
    data["eval"] = json.loads((MODELS / "eval_footfall.json").read_text())
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--daytype", default="mon_thu", choices=list(DT_VEC))
    ap.add_argument("--month", type=int, default=10); ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--export-web", action="store_true")
    a = ap.parse_args()
    if a.export_web:
        export_web()
    elif a.lat is not None:
        p = predict_points(Ensemble(), a.lat, a.lon, a.daytype, month=a.month, year=a.year)[0]
        for h, v in enumerate(p): print(f"{h:02d}:00  {v:8.0f}")
        print(f"daily total ~ {p.sum():,.0f} people")
