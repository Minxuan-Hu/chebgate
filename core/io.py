import csv
import json
from pathlib import Path


def ensure_logdir(logdir: str):
    Path(logdir).mkdir(parents=True, exist_ok=True)


def append_csv_row(csv_path: str, header: list, row: dict):
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
