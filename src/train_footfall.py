"""Train + evaluate the spatio-temporal FootfallNet.

  python train_footfall.py --eval [--fold k]   # 5-fold site-grouped CV + temporal holdout, vs baselines
  python train_footfall.py --final  # train seed ensemble on all data and export to models/
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from model import FootfallNet, SOURCES
import temporal as T

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
torch.set_num_threads(max(1, torch.get_num_threads()))

def load():
    obs, sites = pd.read_parquet(PROC / "obs.parquet"), pd.read_parquet(PROC / "sites.parquet")
    snames = [c for c in sites.columns if c.startswith(("poi_", "road_", "dist_"))]
    idx = sites.reset_index().set_index("site_key")["index"]
    si = idx.loc[obs.site_key].values
    XS = sites[snames].values.astype(np.float32)[si]
    XT = obs[T.NAMES].values.astype(np.float32)
    src = obs.source.map({s: i for i, s in enumerate(SOURCES)}).values
    y = np.log1p(obs.y.values.astype(np.float32))
    n = obs.groupby("site_key").y.transform("size").values
    w = (1 / np.sqrt(n)).astype(np.float32); w /= w.mean()   # each site counts ~sqrt(rows)
    return obs, sites, snames, XS, XT, src, y, w

class Std:
    def fit(self, X): self.m, self.s = X.mean(0), X.std(0) + 1e-6; return self
    def __call__(self, X): return ((X - self.m) / self.s).astype(np.float32)

def fit_net(XS, XT, src, y, w, epochs=12, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    ss, st = Std().fit(XS), Std().fit(XT)
    net = FootfallNet(XS.shape[1], XT.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    tens = [torch.from_numpy(ss(XS)), torch.from_numpy(st(XT)), torch.from_numpy(src).long(),
            torch.from_numpy(y), torch.from_numpy(w)]
    n, bs = len(y), 2048
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    net.train()
    for ep in range(epochs):
        perm, tot = torch.randperm(n), 0.0
        for i in range(n // bs):
            b = perm[i * bs:(i + 1) * bs]
            xs, xt, sr, yy, ww = (t[b] for t in tens)
            xs = xs + 0.05 * torch.randn_like(xs)   # light input noise for spatial robustness
            loss = (ww * torch.nn.functional.huber_loss(net(xs, xt, sr), yy, reduction="none", delta=1.0)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if verbose: print(f"  epoch {ep+1}: loss {tot / (n // bs):.4f}")
    net.eval()
    return net, ss, st

@torch.no_grad()
def predict(model, XS, XT, src):
    net, ss, st = model
    out = [net(torch.from_numpy(ss(XS[i:i+65536])), torch.from_numpy(st(XT[i:i+65536])),
               torch.from_numpy(src[i:i+65536]).long()).numpy() for i in range(0, len(XS), 65536)]
    return np.concatenate(out)

def metrics(obs, pred_log, mask):
    o = obs[mask].assign(p=np.expm1(np.clip(pred_log, 0, 12)), yl=np.log1p(obs.y[mask]), pl=pred_log)
    rmse = float(np.sqrt(((o.yl - o.pl) ** 2).mean()))
    r2 = float(1 - ((o.yl - o.pl) ** 2).sum() / ((o.yl - o.yl.mean()) ** 2).sum())
    # daily totals (surveys: per site-survey; sensors: per site-day)
    key = [o.site_key, o.ts.dt.normalize().fillna(pd.Timestamp(0)), o.year, o.dt_mon_thu]
    d = o.groupby(key)[["y", "p"]].sum()
    d = d[d.y > 0]
    ape = (d.p - d.y).abs() / d.y
    # rank correlation of site means (does the model order locations correctly?)
    sm = o.groupby("site_key")[["y", "p"]].mean()
    rho = float(sm.y.rank().corr(sm.p.rank())) if len(sm) > 2 else float("nan")
    return dict(n=int(mask.sum()), log_rmse=round(rmse, 3), log_r2=round(r2, 3),
                daily_mdape=round(float(ape.median()), 3), within_25pct=round(float((ape < .25).mean()), 3),
                site_rank_spearman=round(rho, 3))

def profile_baseline(obs, y, tr, te):
    """Location-blind baseline: mean log-count per source x daytype x hour."""
    dtc = obs[[f"dt_{d}" for d in T.DAYTYPES]].values.argmax(1)
    key = pd.Series(list(zip(obs.source, dtc, obs.hour)))
    m = pd.Series(y[tr]).groupby(key[tr].values).mean()
    return key[te].map(m).fillna(y[tr].mean()).values

def run_fold(k):
    """Spatial CV fold k (0-4) or k='temporal'. Caches predictions to models/cv/."""
    obs, sites, snames, XS, XT, src, y, w = load()
    obs = obs.reset_index(drop=True)
    out = MODELS / "cv"; out.mkdir(parents=True, exist_ok=True)
    if k == "temporal":
        later = (obs.ts >= "2024-07-01") | (obs.ts.isna() & (obs.year >= (2024.75 - 2020) / 5))
        tr, te = np.where(~later)[0], np.where(later)[0]
        pn = predict(fit_net(XS[tr], XT[tr], src[tr], y[tr], w[tr]), XS[te], XT[te], src[te])
        np.savez(out / "temporal.npz", te=te, net=pn, profile=profile_baseline(obs, y, tr, te))
        return
    tr, te = list(GroupKFold(5).split(XS, y, obs.site_key.values))[k]
    X = np.hstack([XS, XT, np.eye(3)[src]])
    sub = np.random.default_rng(0).choice(tr, min(len(tr), 150_000), replace=False)
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63)
    g.fit(X[sub], y[sub], sample_weight=w[sub])
    np.savez(out / f"fold{k}.npz", te=te, profile=profile_baseline(obs, y, tr, te), gbm=g.predict(X[te]),
             net=predict(fit_net(XS[tr], XT[tr], src[tr], y[tr], w[tr]), XS[te], XT[te], src[te]))

def report():
    obs, *_ , y, w = load()
    obs = obs.reset_index(drop=True)
    S = obs.source.values
    res = {"spatial_cv": {}, "temporal_holdout": {}}
    preds = {k: np.zeros(len(obs)) for k in ("profile", "gbm", "net")}
    for k in range(5):
        f = np.load(MODELS / "cv" / f"fold{k}.npz")
        for name in preds: preds[name][f["te"]] = f[name]
    for name, p in preds.items():
        res["spatial_cv"][name] = {s: metrics(obs, p[S == s], S == s) for s in SOURCES}
    f = np.load(MODELS / "cv" / "temporal.npz")
    m = np.zeros(len(obs), bool); m[f["te"]] = True
    for name in ("profile", "net"):
        p = np.zeros(len(obs)); p[f["te"]] = f[name]
        res["temporal_holdout"][name] = {s: metrics(obs, p[m & (S == s)], m & (S == s)) for s in SOURCES}
    np.save(MODELS / "cv_pred_net.npy", preds["net"].astype(np.float32))
    (MODELS / "eval_footfall.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))

def final(seeds=3, only=None):
    obs, sites, snames, XS, XT, src, y, w = load()
    MODELS.mkdir(exist_ok=True)
    for sd in ([only] if only is not None else range(seeds)):
        net, ss, st = fit_net(XS, XT, src, y, w, epochs=14, seed=sd, verbose=True)
        torch.save({"state_dict": net.state_dict(), "spatial_mean": ss.m, "spatial_std": ss.s,
                    "temporal_mean": st.m, "temporal_std": st.s, "spatial_names": snames,
                    "temporal_names": T.NAMES, "sources": SOURCES}, MODELS / f"footfall_seed{sd}.pt")
        print("saved seed", sd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--fold", default=None, help="0-4 or 'temporal'; omit to run all")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="aggregate cached folds")
    ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    if a.report: report()
    if a.eval:
        folds = [a.fold] if a.fold is not None else [0, 1, 2, 3, 4, "temporal"]
        for k in folds:
            run_fold(int(k) if str(k).isdigit() else k)
        if a.fold is None: report()
    if a.final: final(only=a.seed)
