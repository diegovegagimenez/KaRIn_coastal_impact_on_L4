#!/usr/bin/env python3
"""Regional classification: the transition zone between the North Sea and the
Baltic Sea comprises stations south of 56.44N that lie in the IHO Kattegat
polygon (including the Sound and the Belts) or in the IHO Baltic Sea polygon west
of 13.0E (Kiel and Mecklenburg Bights). The Kattegat north of 56.44N and the
Skagerrak remain in the North West Shelf."""
import os
from pathlib import Path
import pandas as pd, geopandas as gpd
from shapely.geometry import Point
import final_v1_scale_cfg as C
HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "per_station_radius_with_pixels.csv")
sta = D.drop_duplicates("station")[["station", "latitude", "longitude", "region"]].reset_index(drop=True)
gdf = gpd.read_file(os.environ.get("KARIN_IHO_SHP", os.environ.get("KARIN_PROJECT_DIR", "data") + "/World_Seas_IHO_v2/World_Seas_IHO_v2.shp")).to_crs("EPSG:4326")
pts = gpd.GeoDataFrame(sta, geometry=[Point(x, y) for x, y in zip(sta.longitude, sta.latitude)], crs="EPSG:4326")
j = gpd.sjoin_nearest(pts, gdf[["geometry", "NAME"]], how="left"); j = j[~j.index.duplicated()]
sta["iho"] = j.NAME.values
tz = ((sta.iho == "Kattegat") | ((sta.iho == "Baltic Sea") & (sta.longitude < 13.0))) & (sta.latitude <= 56.44)
sta["region_v2"] = sta.region.where(~tz, "Transition zone")
sta.rename(columns={"region": "region_v1"}).to_csv(HERE / "station_regions_v2.csv", index=False)
print(sta[tz].groupby(["iho", "region"]).size())
print(sta.region_v2.value_counts())
