# download_deadtrees.py
"""
Download packages from deadtrees.earth.
Since the image tiles package is 287 GB, we stream it remotely using HTTP Range requests
and extract only a representative subset of 100 image tiles to prevent running out of disk space.
"""
import os
import io
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

OUT_DIR = "DeadTrees/raw"
os.makedirs(OUT_DIR, exist_ok=True)

DOWNLOADS = {
    "standing-deadwood-aerial-global-conservative.zip":
        "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/standing-deadwood-aerial-global-conservative_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22standing-deadwood-aerial-global-conservative_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260714%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260714T145518Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0cfc0ccf0628e7f8eb33729a90eae757eeac149f57ecce7583dd43109f816f26",

    "tree-cover-aerial-global.zip":
        "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/tree-cover-aerial-global_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22tree-cover-aerial-global_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260714%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260714T145555Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6951db7cb6f4dac073b61ac53dcd2e41c7b8af61263c37067ecb35b624e8749a",
}

AERIAL_URL = "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260916%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260916T062515Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=808c0091d3299b1c1a639acc1a8c04e28e0168d88093bf4be04dd6631f65ede0"

SCALEUP_SELECTION_PATH = os.path.join("DeadTrees", "raw", "scaleup_selection_v1.json")

class HTTPRangeFile:
    def __init__(self, url, buffer_size=8 * 1024 * 1024):
        self.url = url
        self.offset = 0
        r = requests.get(url, headers={'Accept-Encoding': 'identity'}, stream=True)
        self.size = int(r.headers.get('Content-Length', 0))
        r.close()
        self.buffer_size = buffer_size
        self.buffer = b""
        self.buffer_start = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = self.size + offset
        return self.offset

    def tell(self):
        return self.offset

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.offset
            
        if self.offset >= self.size:
            return b""
            
        buf_offset = self.offset - self.buffer_start
        if 0 <= buf_offset < len(self.buffer):
            n = min(size, len(self.buffer) - buf_offset)
            data = self.buffer[buf_offset:buf_offset + n]
            self.offset += n
            if len(data) == size:
                return data
            remaining = size - len(data)
            return data + self.read(remaining)

        fetch_size = max(self.buffer_size, size)
        end = min(self.offset + fetch_size - 1, self.size - 1)
        headers = {
            "Range": f"bytes={self.offset}-{end}",
            "Accept-Encoding": "identity"
        }
        r = requests.get(self.url, headers=headers)
        if r.status_code not in (200, 206):
            raise IOError(f"HTTP Range request failed: {r.status_code}")
            
        self.buffer = r.content
        self.buffer_start = self.offset
        
        n = min(size, len(self.buffer))
        data = self.buffer[:n]
        self.offset += n
        return data


def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return
    response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))

    with open(out_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, desc=os.path.basename(out_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  Extracted -> {extract_to}")


def main():
    # 1. Download & Extract standing-deadwood and tree-cover packages (small packages)
    for filename, url in DOWNLOADS.items():
        extract_dir = os.path.join(OUT_DIR, filename.replace(".zip", ""))
        # Check if already extracted
        if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 1:
            print(f"  {filename} already extracted to {extract_dir}, skipping.")
            continue

        out_path = os.path.join(OUT_DIR, filename)
        print(f"Downloading {filename}...")
        download_file(url, out_path)

        os.makedirs(extract_dir, exist_ok=True)
        extract_zip(out_path, extract_dir)
        
        # Delete zip file to free disk space
        if os.path.exists(out_path):
            print(f"  Deleting raw zip: {out_path}")
            os.remove(out_path)

    # 2. Extract subset of 100 tiles from 287GB aerial image zip file remotely
    aerial_extract_dir = os.path.join(OUT_DIR, "image-tiles-1024-global-aerial-sampled-20-random")
    os.makedirs(aerial_extract_dir, exist_ok=True)

    # Target dataset ids that we want to download tiles for. Base 5 sites are
    # kept for backward compatibility; if a biome-stratified scale-up
    # selection exists (select_scaleup_datasets.py), its sites are added too.
    target_datasets = [3889, 3968, 5737, 5653, 5650]
    if os.path.exists(SCALEUP_SELECTION_PATH):
        import json
        with open(SCALEUP_SELECTION_PATH) as fh:
            scaleup = json.load(fh)
        target_datasets = sorted(set(target_datasets) | set(scaleup["selected_dataset_ids"]))
        print(f"  Loaded scale-up selection: {len(scaleup['selected_dataset_ids'])} extra sites "
              f"({len(target_datasets)} total) from {SCALEUP_SELECTION_PATH}")
    max_tiles_per_dataset = 20

    # Count how many tifs exist already
    import glob
    existing_tifs = glob.glob(f"{aerial_extract_dir}/**/*.tif", recursive=True)
    expected_tifs = len(target_datasets) * max_tiles_per_dataset
    if len(existing_tifs) >= expected_tifs:
        print(f"  Already extracted {len(existing_tifs)}/{expected_tifs} image tiles, skipping remote extraction.")
        return

    print("Opening 287 GB remote zip file to list tiles by dataset_id (directory listing only)...")
    with zipfile.ZipFile(HTTPRangeFile(AERIAL_URL)) as z:
        namelist = z.namelist()

    # Group tiles by dataset_id
    tiles_by_dataset = {d: [] for d in target_datasets}
    for name in namelist:
        if not name.endswith(".tif"):
            continue
        # Path format: tiles/{dataset_id}/dataset_{dataset_id}_r{row}_c{col}.tif
        parts = name.split("/")
        if len(parts) >= 3 and parts[0] == "tiles":
            try:
                dataset_id = int(parts[1])
                if dataset_id in tiles_by_dataset:
                    tiles_by_dataset[dataset_id].append(name)
            except ValueError:
                continue

    # Each worker opens its own HTTPRangeFile + ZipFile: HTTPRangeFile keeps
    # mutable offset/buffer state, so sharing one instance across threads
    # would corrupt reads. One ZipFile per dataset_id keeps the per-worker
    # central-directory reparse cost amortized over that dataset's 20 tiles.
    def download_one_dataset(dataset_id, name_list):
        selected = name_list[:max_tiles_per_dataset]
        pending = [n for n in selected if not os.path.exists(os.path.join(aerial_extract_dir, n))]
        if not pending:
            return dataset_id, 0
        with zipfile.ZipFile(HTTPRangeFile(AERIAL_URL)) as z:
            count = 0
            for name in pending:
                dest_path = os.path.join(aerial_extract_dir, name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with z.open(name) as tile_f, open(dest_path, "wb") as out_tile_f:
                    out_tile_f.write(tile_f.read())
                count += 1
        return dataset_id, count

    from concurrent.futures import ThreadPoolExecutor, as_completed
    extracted_count = 0
    jobs = {d: names for d, names in tiles_by_dataset.items()}
    print(f"Downloading tiles for {len(jobs)} datasets with 12 parallel workers...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(download_one_dataset, d, names): d for d, names in jobs.items()}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                dataset_id, count = future.result()
                extracted_count += count
                pbar.update(1)
                pbar.set_postfix(last_dataset=dataset_id, new_tiles=count)

    print(f"Successfully extracted {extracted_count} target image tiles.")

if __name__ == "__main__":
    main()
