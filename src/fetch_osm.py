"""Fetch OpenStreetMap POIs, transit stops and the street network for each study area."""
import json, sys, time, urllib.request, urllib.parse
from pathlib import Path

MIRRORS = ["https://maps.mail.ru/osm/tools/overpass/api/interpreter",
           "https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]
AREAS = {  # south, west, north, east
    "sydney_cbd": (-33.935, 151.165, -33.850, 151.240),
    "liverpool":  (-33.940, 150.905, -33.910, 150.940),
}
POI_Q = """[out:json][timeout:180];
(
  nwr["shop"]({b}); nwr["amenity"]({b}); nwr["office"]({b}); nwr["tourism"]({b});
  nwr["leisure"]({b}); nwr["railway"~"station|tram_stop|subway_entrance|halt"]({b});
  nwr["public_transport"="station"]({b}); node["highway"="bus_stop"]({b});
  nwr["building"~"university|college|school|hospital|stadium|train_station"]({b});
);
out tags center;"""
ROAD_Q = """[out:json][timeout:180];
way["highway"~"^(primary|secondary|tertiary|residential|unclassified|pedestrian|footway|living_street|trunk|service|steps|path|cycleway)$"]({b});
out tags geom;"""

def overpass(q, rounds=4):
  for r in range(rounds):
    for m in MIRRORS:
      try:
            req = urllib.request.Request(m, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "sydney-pedestrian-model/0.1"})
            return json.load(urllib.request.urlopen(req, timeout=240))
      except Exception as e:
            print("  mirror failed", m, e, file=sys.stderr); time.sleep(10 * (r + 1))
  raise RuntimeError("all Overpass mirrors failed")

out = Path(__file__).resolve().parents[1] / "data" / "raw" / "osm"
out.mkdir(parents=True, exist_ok=True)
for name, (s, w, n, e) in AREAS.items():
    b = f"{s},{w},{n},{e}"
    for kind, q in (("poi", POI_Q), ("roads", ROAD_Q)):
        if (out / f"{name}_{kind}.json").exists():
            continue  # cached
        d = overpass(q.format(b=b))
        (out / f"{name}_{kind}.json").write_text(json.dumps(d["elements"]))
        print(name, kind, len(d["elements"]))
