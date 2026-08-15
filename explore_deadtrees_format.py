# explore_deadtrees_format.py
"""
Kham pha cau truc data sau khi giai nen 3 packages.
KHONG doan format -- luon verify truoc khi viet code xu ly tiep.
"""
import glob, os

print("=" * 60)
print("LIET KE TOAN BO FILE TRONG 3 THU MUC DA GIAI NEN")
print("=" * 60)

folders = {
    "deadwood":   "DeadTrees/raw/standing-deadwood-aerial-global-conservative",
    "treecover":  "DeadTrees/raw/tree-cover-aerial-global",
    "aerial":     "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random",
}

found_files = {}
for name, folder in folders.items():
    all_files = glob.glob(f"{folder}/**/*", recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    found_files[name] = all_files
    print(f"\n[{name}] {len(all_files)} files")
    exts = set(os.path.splitext(f)[1] for f in all_files)
    print(f"  Extensions: {exts}")
    for f in all_files[:5]:
        print(f"  - {f}")

print("\n" + "=" * 60)
print("2. KIEM TRA FILE VECTOR (GEOPACKAGE/SHAPEFILE/GEOJSON)")
print("=" * 60)
import geopandas as gpd

for name in ["deadwood", "treecover"]:
    vector_files = [f for f in found_files[name]
                    if f.endswith((".gpkg", ".shp", ".geojson"))]
    for vf in vector_files:
        print(f"\n[{name}] {vf}")
        gdf = gpd.read_file(vf)
        print(f"  Rows: {len(gdf)}")
        print(f"  CRS: {gdf.crs}")
        print(f"  Columns: {list(gdf.columns)}")
        id_cols = [c for c in gdf.columns if any(
            k in c.lower() for k in ['id', 'image', 'file', 'ortho', 'source'])]
        print(f"  Cot co the lien ket voi anh: {id_cols}")
        if len(gdf) > 0:
            print(f"  Sample row:\n{gdf.iloc[0]}")

print("\n" + "=" * 60)
print("3. KIEM TRA FILE ANH (TIF/PNG)")
print("=" * 60)
import rasterio

img_files = [f for f in found_files["aerial"]
             if f.endswith((".tif", ".tiff", ".png", ".jpg"))]
if img_files:
    sample = img_files[0]
    print(f"Sample: {sample}")
    if sample.endswith((".tif", ".tiff")):
        with rasterio.open(sample) as src:
            print(f"  CRS: {src.crs}")
            print(f"  Size: {src.width} x {src.height}")
            print(f"  Bands: {src.count}, Dtype: {src.dtypes}")
            print(f"  Bounds: {src.bounds}")
    else:
        from PIL import Image
        img = Image.open(sample)
        print(f"  Size: {img.size}, Mode: {img.mode}")
        print(f"  [WARNING] .png/.jpg khong co geospatial metadata")
        print(f"  Can tim file world file (.tfw/.wld) hoac metadata rieng")

print("\n[AGENT DECISION]")
print("Xac dinh: moi polygon lien ket voi anh nao qua cot gi?")
print("CRS cua anh va polygon co khop nhau khong?")
print("Anh co geospatial metadata (CRS) hay chi la RGB thuan?")
