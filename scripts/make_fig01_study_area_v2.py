#!/usr/bin/env python3
"""Fig. 1 (v3): area de estudio y clasificacion regional v2 (7 regiones, incluida
la 'Transition zone' = estaciones al sur de 56.44N en IHO Kattegat [Sound y Belts] o
en IHO Baltic Sea al oeste de 13E). Poligonos IHO coloreados por region y estaciones."""
import os
from pathlib import Path
import pandas as pd, geopandas as gpd
from shapely.geometry import box
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs, cartopy.feature as cfeature
import final_v1_scale_cfg as C
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "figures"
COL = {"Transition zone": "#e31a1c", "North West Shelf": "#d9a441", "Baltic Sea": "#3f9f8f",
       "Iberia-Biscay-Ireland": "#9c5a52", "Mediterranean Sea": "#4f7d9c", "Arctic": "#8a8a8a",
       "Black Sea": "#7b5a8e"}
g = gpd.read_file(os.environ.get("KARIN_IHO_SHP", os.environ.get("KARIN_PROJECT_DIR", "data") + "/World_Seas_IHO_v2/World_Seas_IHO_v2.shp")).to_crs("EPSG:4326")
g["region"] = g.NAME.map(C.RMAP); g = g.dropna(subset=["region"]).copy()
south = box(-180, -90, 180, 56.44)
k = g[g.NAME == "Kattegat"]
ks = k.copy(); ks["geometry"] = k.geometry.intersection(south); ks["region"] = "Transition zone"
kn = k.copy(); kn["geometry"] = k.geometry.difference(south)          # sigue en North West Shelf
west = box(-180, -90, 13.0, 56.44)
b = g[g.NAME == "Baltic Sea"]
bw = b.copy(); bw["geometry"] = b.geometry.intersection(west); bw["region"] = "Transition zone"
be = b.copy(); be["geometry"] = b.geometry.difference(west)
g = pd.concat([g[~g.NAME.isin(["Baltic Sea", "Kattegat"])], ks, kn, bw, be])
g = g.clip(box(-13, 30, 43, 72))
st = pd.read_csv(HERE / "station_regions_v2.csv")
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8.5, 9)); ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([-12, 42, 31, 68.5], crs=proj)
ax.add_feature(cfeature.LAND, facecolor="#d5d8de", zorder=1)
for reg, gg in g.groupby("region"):
    ax.add_geometries(gg.geometry, crs=proj, facecolor=COL[reg], edgecolor="none", alpha=0.9, zorder=2)
ax.add_feature(cfeature.LAND, facecolor="#d5d8de", zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
ax.scatter(st.longitude, st.latitude, s=7, c="k", transform=proj, zorder=5)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5); gl.top_labels = gl.right_labels = False
n = st.region_v2.value_counts()
h = [Patch(facecolor=COL[r], label=f"{r} ({n.get(r, 0)})") for r in COL]
h.append(Line2D([], [], marker="o", color="k", ls="", ms=3, label="Tide gauge stations"))
ax.legend(handles=h, loc="upper left", fontsize=8, framealpha=0.95)
# inset de la zona de transicion
axi = fig.add_axes([0.625, 0.395, 0.225, 0.20], projection=ccrs.Mercator())
axi.set_extent([8.8, 14.2, 53.6, 57.2], crs=proj)
for reg, gg in g.groupby("region"):
    axi.add_geometries(gg.geometry, crs=proj, facecolor=COL[reg], edgecolor="none", alpha=0.9, zorder=2)
axi.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#d5d8de", zorder=3)
axi.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.4, zorder=4)
axi.scatter(st.longitude, st.latitude, s=9, c="k", transform=proj, zorder=5)
axi.set_title("Transition zone", fontsize=8.5, pad=2)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig01_study_area_map_v2.{ext}", dpi=250, bbox_inches="tight")
print("OK", n.to_dict())
