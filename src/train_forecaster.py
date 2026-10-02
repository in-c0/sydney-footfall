"""Next-24-hour forecaster for continuous counters (CoS sensors + Liverpool cameras).

Each sample: origin = local midnight; input = previous 168 hourly log-counts; target = next 24.
Split by time: last 20% of each site's days are the test set.
Baselines: seasonal-naive (same hour, one week earlier) and 4-week same-weekday mean.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import ShortTermForecaster
import temporal as T

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
FUT = [n for n in T.NAMES if not n.startswith("h_")] + ["src_cos", "src_liv"]

def samples():
    obs = pd.read_parquet(PROC / "obs.parquet")
    obs = obs[obs.source != "cos_survey"]
    P, PH, F, Y, meta = [], [], [], [], []
    for key, g in obs.groupby("site_key"):
        g = g.set_index("ts").sort_index()
        idx = pd.date_range(g.index.min().normalize(), g.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
        g = g.reindex(idx)
        ly = np.log1p(g.y.values)
        feat = g[[n for n in T.NAMES if not n.startswith("h_")]].values
        srcf = np.tile([key.startswith("cos"), key.startswith("liverpool")], (len(g), 1))
        feat = np.hstack([feat, srcf]).astype(np.float32)
        days = np.arange(7, len(idx) // 24)
        for d in days:
            past, fut = ly[(d - 7) * 24:d * 24], ly[d * 24:(d + 1) * 24]
            if np.isnan(past).sum() > 12 or np.isnan(fut).any(): continue
            P.append(pd.Series(past).ffill().bfill().values); PH.append(np.tile(np.arange(24), 7))
            F.append(feat[d * 24:(d + 1) * 24]); Y.append(fut); meta.append((key, idx[d * 24]))
    P, PH, F, Y = (np.array(a, np.float32) for a in (P, PH, F, Y))
    m = pd.DataFrame(meta, columns=["site", "day"])
    m["test"] = m.groupby("site").day.rank(pct=True) > 0.8
    return P, PH, np.nan_to_num(F), Y, m

def main(epochs=12):  # noqa: C901
    cache = PROC / "forecast_samples.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        P, PH, F, Y = z["P"], z["PH"], z["F"], z["Y"]
        m = pd.DataFrame({"site": z["site"], "day": z["day"], "test": z["test"]})
    else:
        P, PH, F, Y, m = samples()
        np.savez(cache, P=P, PH=PH, F=F, Y=Y, site=m.site.values, day=m.day.values, test=m.test.values)
    tr, te = ~m.test.values, m.test.values
    fm, fs = F[tr].reshape(-1, F.shape[-1]).mean(0), F[tr].reshape(-1, F.shape[-1]).std(0) + 1e-6
    Fn = (F - fm) / fs
    torch.manual_seed(0)
    net = ShortTermForecaster(F.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    tP, tH, tF, tY = (torch.from_numpy(np.ascontiguousarray(a).copy()) for a in (P, PH.astype(np.float32), Fn.astype(np.float32), Y))
    itr = np.where(tr)[0]
    for ep in range(epochs):
        np.random.shuffle(itr); tot = 0
        net.train()
        for i in range(0, len(itr), 256):
            b = torch.from_numpy(itr[i:i + 256])
            loss = torch.nn.functional.huber_loss(net(tP[b], tH[b], tF[b]), tY[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        print(f"epoch {ep+1} train huber {tot/len(itr):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        pred = net(tP[te], tH[te], tF[te]).numpy()
    yt = Y[te]
    naive = P[te][:, :24]                     # same hours one week earlier
    def score(p):
        a, b = np.expm1(yt).sum(1), np.expm1(np.clip(p, 0, 12)).sum(1)
        ok = a > 0
        return dict(hourly_log_rmse=round(float(np.sqrt(((p - yt) ** 2).mean())), 3),
                    daily_mdape=round(float(np.median(np.abs(b[ok] - a[ok]) / a[ok])), 3))
    res = {"n_train_days": int(tr.sum()), "n_test_days": int(te.sum()),
           "seasonal_naive_1w": score(naive), "gru_forecaster": score(pred)}
    print(json.dumps(res, indent=1))
    MODELS.mkdir(exist_ok=True)
    (MODELS / "eval_forecaster.json").write_text(json.dumps(res, indent=2))
    torch.save({"state_dict": net.state_dict(), "fut_mean": fm, "fut_std": fs, "fut_names": FUT},
               MODELS / "forecaster.pt")

if __name__ == "__main__":
    main()
