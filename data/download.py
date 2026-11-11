"""Download every source in sources.csv that has a direct_pdf_url into data/raw/.

Usage: python data/download.py
Rows without a direct_pdf_url must be downloaded manually from landing_url
(save as data/raw/<filename>); the script reports which files are missing.
"""
import csv
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)

with open(HERE / "sources.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

missing = []
for row in rows:
    target = RAW / row["filename"]
    if target.exists():
        print(f"ok       {row['filename']}")
        continue
    url = row["direct_pdf_url"].strip()
    if not url:
        missing.append(row)
        continue
    req = urllib.request.Request(url, headers={"User-Agent": "rag-assistant/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        target.write_bytes(resp.read())
    print(f"download {row['filename']}")

if missing:
    print("\nDownload manually (save to data/raw/):")
    for row in missing:
        print(f"  {row['filename']:<14} <- {row['landing_url']}")
