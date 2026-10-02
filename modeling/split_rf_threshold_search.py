"""Losslessly partition existing RF search exports without fitting any models.

Run: python modeling/split_rf_threshold_search.py cn7 rg3
Original single-file CSVs remain local backups. Generated files are the two
parts and their integrity manifest; no numeric text or row order is changed.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def split_search(folder: Path) -> dict:
    source = folder / "threshold_search.csv"
    # These model exports contain no multiline fields. Check this assumption
    # before partitioning the raw bytes (never reserialize floating-point text).
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        row_count = 0
        for row in reader:
            if len(row) != len(columns) or any("\n" in value or "\r" in value for value in row):
                raise ValueError("Unexpected multiline or malformed CSV record")
            row_count += 1
    if row_count != 7 * 108 * 1001:
        raise ValueError(f"Unexpected row count: {row_count}")
    midpoint = (row_count + 1) // 2
    parts = []
    with source.open("rb") as stream:
        header = stream.readline()
        for index, count in enumerate((midpoint, row_count - midpoint), 1):
            path = folder / f"threshold_search_part{index}.csv"
            # Refuse to overwrite a possibly user-modified split export.
            with path.open("xb") as target:
                target.write(header)
                written = 0
                for line in itertools.islice(stream, count):
                    target.write(line)
                    written += 1
            assert written == count
            assert path.stat().st_size < 100_000_000, path
            parts.append(dict(file=path.name, rows=count, bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert not stream.read()
    # Rejoining part1 with part2 (excluding the repeated header) must reproduce
    # the original file byte for byte, not merely rounded numeric equivalents.
    restored = hashlib.sha256()
    for index, record in enumerate(parts):
        with (folder / record["file"]).open("rb") as stream:
            if index:
                assert stream.readline() == header
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                restored.update(block)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert restored.hexdigest() == source_hash
    manifest = dict(format_version=1, total_rows=row_count, columns=columns,
                    parts=parts, source_sha256=source_hash,
                    byte_identical_reassembly=True)
    (folder / "threshold_search_parts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=("cn7", "rg3"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for dataset in args.datasets:
        folder = root / "output" / f"random_forest_{dataset}" / "manual_seven_scenarios_v1"
        print(json.dumps(dict(dataset=dataset, **split_search(folder)), ensure_ascii=False), flush=True)
