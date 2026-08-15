# find_optimal_datasets.py
import geopandas as gpd
import zipfile
import requests
from download_deadtrees import HTTPRangeFile

url = "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260714%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260714T145545Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=327412e71c409484c48662ce609a50349bc8a3d071ca38c181bdd999ede89890"

print("Loading standing_deadwood polygons...")
gdf = gpd.read_file('DeadTrees/raw/standing-deadwood-aerial-global-conservative/standing-deadwood-aerial-global-conservative_2026.06.17.gpkg', layer='standing_deadwood')
counts = gdf['dataset_id'].value_counts()
top_datasets = list(counts.index)

print("Opening remote zip...")
f = HTTPRangeFile(url)
with zipfile.ZipFile(f) as z:
    namelist = z.namelist()
    zip_counts = {}
    for name in namelist:
        if name.endswith(".tif"):
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "tiles":
                try:
                    ds = int(parts[1])
                    zip_counts[ds] = zip_counts.get(ds, 0) + 1
                except ValueError:
                    continue

# Find the top datasets by polygon count that also have 20 tiles in the zip
optimal_datasets = []
for ds in top_datasets:
    if zip_counts.get(ds, 0) == 20:
        optimal_datasets.append((ds, counts[ds]))
        if len(optimal_datasets) == 5:
            break

print("\nTop 5 optimal datasets:")
for ds, poly_cnt in optimal_datasets:
    print(f"  Dataset ID {ds}: {poly_cnt} deadwood polygons, 20 tiles in zip")
