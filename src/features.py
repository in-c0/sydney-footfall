"""Spatial context features from OpenStreetMap, shared by training sites and prediction points."""
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import KDTree

OSM = Path(__file__).resolve().parents[1] / "data" / "raw" / "osm"
LAT0 = -33.88
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))
RADII = (100, 250, 500)

def to_xy(lat, lon):
    return np.c_[(np.asarray(lon) - 151.0) * M_PER_DEG_LON, (np.asarray(lat) - LAT0) * M_PER_DEG_LAT]

FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "biergarten", "nightclub"}
CIVIC = {"library", "townhall", "community_centre", "theatre", "cinema", "arts_centre", "place_of_worship", "museum"}
EDU = {"university", "college", "school"}

def poi_category(t):
    a = t.get("amenity")
    if a in FOOD: return "food"
    if "shop" in t: return "retail"
    if "office" in t: return "office"
    if t.get("railway") == "tram_stop" or t.get("station") == "light_rail" \
            or (t.get("light_rail") == "yes" and t.get("train") != "yes"):
        return "lightrail"
    if t.get("railway") in ("station", "subway_entrance", "halt") or t.get("public_transport") == "station" \
            or t.get("building") == "train_station":
        return "rail"
    if t.get("highway") == "bus_stop": return "bus"
    if a in EDU or t.get("building") in EDU: return "education"
    if "tourism" in t or a in CIVIC: return "culture"
    if t.get("leisure") in ("park", "garden", "playground", "sports_centre", "fitness_centre", "stadium"):
        return "leisure"
    if a: return "other_amenity"
    return None

ROAD_CLASS = {"pedestrian": "ped", "footway": "ped", "living_street": "ped", "steps": "ped", "path": "ped",
              "primary": "major", "secondary": "major", "trunk": "major",
              "tertiary": "minor", "residential": "minor", "unclassified": "minor", "service": "service"}
POI_CATS = ["food", "retail", "office", "rail", "lightrail", "bus", "education", "culture", "leisure", "other_amenity"]
ROAD_CATS = ["ped", "major", "minor", "service"]

class SpatialFeaturizer:
    def __init__(self):
        pts = {c: [] for c in POI_CATS}
        road = {c: [] for c in ROAD_CATS}
        for f in sorted(OSM.glob("*_poi.json")):
            for e in json.loads(f.read_text()):
                c = poi_category(e.get("tags", {}))
                if not c: continue
                lat, lon = (e["lat"], e["lon"]) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
                pts[c].append((lat, lon))
        for f in sorted(OSM.glob("*_roads.json")):
            for e in json.loads(f.read_text()):
                rc = ROAD_CLASS.get(e.get("tags", {}).get("highway"))
                g = e.get("geometry")
                if not rc or not g: continue
                xy = to_xy([p["lat"] for p in g], [p["lon"] for p in g])
                for a, b in zip(xy[:-1], xy[1:]):  # resample every ~10 m
                    n = max(1, int(np.hypot(*(b - a)) // 10))
                    t = (np.arange(n) + 0.5)[:, None] / n
                    road[rc].append(a + t * (b - a))
        self.poi_trees = {c: KDTree(to_xy(*np.array(v).T)) for c, v in pts.items() if v}
        self.road_trees = {c: KDTree(np.vstack(v)) for c, v in road.items() if v}
        self.names = ([f"poi_{c}_{r}" for c in POI_CATS for r in RADII] +
                      [f"road_{c}_{r}" for c in ROAD_CATS for r in RADII] +
                      ["dist_rail", "dist_lightrail", "dist_bus", "dist_ped"])

    def __call__(self, lat, lon):
        xy = to_xy(lat, lon)
        cols = []
        for c in POI_CATS:
            for r in RADII:
                t = self.poi_trees.get(c)
                cols.append(np.log1p(t.query_radius(xy, r, count_only=True)) if t else np.zeros(len(xy)))
        for c in ROAD_CATS:
            for r in RADII:  # metres of street of this class within r (10 m samples)
                t = self.road_trees.get(c)
                cols.append(np.log1p(10 * t.query_radius(xy, r, count_only=True)) if t else np.zeros(len(xy)))
        for c, t in (("rail", self.poi_trees.get("rail")), ("lightrail", self.poi_trees.get("lightrail")),
                     ("bus", self.poi_trees.get("bus")), ("ped", self.road_trees.get("ped"))):
            d = t.query(xy, k=1)[0][:, 0] if t else np.full(len(xy), 5000.0)
            cols.append(np.log1p(np.minimum(d, 5000)))
        return np.column_stack(cols).astype(np.float32)
