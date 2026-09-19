#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

log_dir="reports"
mkdir -p "$log_dir" benchmark/manifests benchmark/qc
log="$log_dir/g0b_background.log"
exec >>"$log" 2>&1

bam="data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"
quebec="data/itc_benchmarks/raw_archives/quebec_trees_dataset_2021-09-02.zip"
bci="data/itc_benchmarks/raw_archives/BCI_50ha_2020_08_01_crownmap_raw.zip"

wait_for_archive() {
    local path="$1"
    local expected="$2"
    local marker="$3"
    while true; do
        local actual=0
        if [[ -f "$path" ]]; then
            actual=$(stat -c %s "$path")
        fi
        if [[ "$actual" -eq "$expected" ]]; then
            echo "READY $marker $actual bytes $(date --iso-8601=seconds)"
            return 0
        fi
        if [[ "$actual" -gt "$expected" ]]; then
            echo "ERROR $marker is larger than frozen source size: $actual > $expected"
            return 1
        fi
        if ! pgrep -f "curl.*${path##*/}" >/dev/null; then
            echo "ERROR $marker download stopped at $actual/$expected bytes; not auto-restarting"
            return 1
        fi
        echo "WAIT $marker $actual/$expected bytes $(date --iso-8601=seconds)"
        sleep 60
    done
}

echo "G0B background finalizer started $(date --iso-8601=seconds)"
wait_for_archive "$bam" 30410910782 BAMFORESTS
wait_for_archive "$quebec" 19050475430 QUEBEC

[[ "$(stat -c %s "$bci")" -eq 2817926041 ]]

echo "Validating official MD5 checksums"
echo "686fde075eb5a81f48cf769a2599b488  $bci" | md5sum -c -
echo "c0ad1db2a7aa3c20ad577d47724dbe4e  $quebec" | md5sum -c -

echo "Testing ZIP structure"
unzip -tq "$bam"
unzip -tq "$quebec"
unzip -tq "$bci"

echo "Writing digests"
{
    md5sum "$bam" "$quebec" "$bci"
    sha256sum "$bam" "$quebec" "$bci"
} > reports/g0b_checksums.txt

echo "Selectively extracting annotations and three Sep-02 RGB COGs"
mkdir -p data/itc_benchmarks/extracted/bam_coco2048 data/itc_benchmarks/extracted/quebec_2021_09_02
while IFS= read -r member; do
    unzip -j -n "$bam" "$member" -d data/itc_benchmarks/extracted/bam_coco2048
done < <(unzip -Z1 "$bam" | awk 'tolower($0) ~ /\.json$/ || tolower($0) ~ /readme/')
while IFS= read -r member; do
    unzip -j -n "$quebec" "$member" -d data/itc_benchmarks/extracted/quebec_2021_09_02
done < <(unzip -Z1 "$quebec" | awk 'tolower($0) ~ /2021-09-02-sbl-z[123]-rgb-cog\.tif$/')

[[ "$(find data/itc_benchmarks/extracted/bam_coco2048 -maxdepth 1 -name '*.json' | wc -l)" -eq 4 ]]
[[ "$(find data/itc_benchmarks/extracted/quebec_2021_09_02 -maxdepth 1 -name '*rgb-cog.tif' | wc -l)" -eq 3 ]]

chmod 0444 "$bam" "$quebec" "$bci"
find data/itc_benchmarks/extracted/bam_coco2048 -type f -exec chmod 0444 {} +
find data/itc_benchmarks/extracted/quebec_2021_09_02 -type f -exec chmod 0444 {} +
find data/itc_benchmarks/extracted/bci_2021_raw -type f -exec chmod 0444 {} +
if find \
    data/itc_benchmarks/raw_archives \
    data/itc_benchmarks/extracted/bam_coco2048 \
    data/itc_benchmarks/extracted/quebec_2021_09_02 \
    data/itc_benchmarks/extracted/bci_2021_raw \
    -type f -perm /222 -print -quit | grep -q .; then
    echo "ERROR source data is still writable"
    exit 1
fi

echo "Running full G0B geometry and visual QC"
python scripts/g0b_data_qc.py --dataset all
python scripts/g0b_write_report.py

echo "Artifact inventory"
find benchmark/manifests benchmark/qc -maxdepth 1 -type f -printf '%p %s bytes\n' | sort
ls -l benchmark/annotation_policy.md reports/g0b_data_qc_report.md
echo "G0B background finalizer complete $(date --iso-8601=seconds)"
