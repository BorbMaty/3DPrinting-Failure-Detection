#!/usr/bin/env python3
"""
image_downloader.py
Tömeges képletöltő script:
- mode=urls : beolvassa az URL-eket (txt vagy csv) és letölti
- mode=crawl: beolvassa a megadott weboldalt, kigyűjti az <img> src-ket majd letölti

Példa:
python image_downloader.py --mode urls --input urls.txt --outdir dataset/
python image_downloader.py --mode crawl --input https://example.com --outdir dataset/
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import async_timeout
import csv
import hashlib
import os
import re
import sys
from urllib.parse import urljoin, urlparse
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

# ----- Helper functions -----
def safe_filename_from_url(url: str, prefix: str = "") -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name or "." not in name:
        # no filename: use hash + ext fallback
        h = hashlib.sha1(url.encode()).hexdigest()[:10]
        ext = ".jpg"
        name = f"{h}{ext}"
    # sanitize
    name = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    if prefix:
        name = f"{prefix}_{name}"
    return name

async def fetch_html(session: aiohttp.ClientSession, url: str, timeout: int = 15):
    try:
        async with async_timeout.timeout(timeout):
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.text()
    except Exception as e:
        print(f"[crawl] Error fetching {url}: {e}")
        return None

def extract_img_urls(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    urls = set()
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        # resolve relative URLs
        full = urljoin(base_url, src)
        urls.add(full)
    return list(urls)

# ----- Main downloader -----
async def download_image(session: aiohttp.ClientSession, url: str, outpath: Path,
                         sem: asyncio.Semaphore, retries: int = 2, timeout: int = 20,
                         headers=None):
    headers = headers or {}
    last_exc = None
    async with sem:
        for attempt in range(retries + 1):
            try:
                async with async_timeout.timeout(timeout):
                    async with session.get(url, headers=headers) as resp:
                        if resp.status != 200:
                            raise aiohttp.ClientError(f"Status {resp.status}")
                        # stream to file
                        tmp_path = outpath.with_suffix(outpath.suffix + ".part")
                        async with aiofiles.open(tmp_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(1024):
                                await f.write(chunk)
                        # rename atomically
                        tmp_path.rename(outpath)
                        return {"url": url, "file": str(outpath), "status": "ok", "size": outpath.stat().st_size}
            except Exception as e:
                last_exc = e
                # small backoff
                await asyncio.sleep(0.5 + attempt * 0.8)
        return {"url": url, "file": None, "status": f"failed: {last_exc}", "size": 0}

async def run_downloader(urls, outdir: Path, concurrency=8, retries=2, timeout=20, prefix="img"):
    outdir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    # headers - set common user-agent to avoid blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ImageDatasetBot/1.0; +https://example.com/bot)"
    }
    connector = aiohttp.TCPConnector(limit=concurrency*2)
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        tasks = []
        seen_hashes = set()
        results = []
        pbar = tqdm(total=len(urls), desc="Downloading", unit="img")
        for i, url in enumerate(urls):
            # determine output filename, avoid collisions
            filename = safe_filename_from_url(url, prefix)
            outpath = outdir / filename
            # if exists, try to append an index
            idx = 1
            while outpath.exists():
                # optional: check hash to skip identical file (not implemented here)
                outpath = outdir / f"{outpath.stem}_{idx}{outpath.suffix}"
                idx += 1
            task = asyncio.create_task(download_image(session, url, outpath, sem,
                                                      retries=retries, timeout=timeout, headers=headers))
            task.add_done_callback(lambda t: pbar.update(1))
            tasks.append(task)
        for fut in asyncio.as_completed(tasks):
            r = await fut
            results.append(r)
        pbar.close()
    return results

# ----- Input parsing helpers -----
def read_urls_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return urls

def read_urls_from_csv(path, colname="url"):
    urls = []
    with open(path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if colname not in reader.fieldnames:
            raise ValueError(f"Column '{colname}' not found in CSV. Columns: {reader.fieldnames}")
        for row in reader:
            u = row.get(colname)
            if u:
                urls.append(u.strip())
    return urls

# ----- CLI -----
def parse_args():
    p = argparse.ArgumentParser(description="Mass image downloader for datasets")
    p.add_argument("--mode", choices=["urls", "crawl"], required=True, help="urls = read from input list; crawl = scrape page for <img> tags")
    p.add_argument("--input", required=True, help="path to input file (txt/csv) or URL (for crawl mode)")
    p.add_argument("--incol", default="url", help="when input is csv, column name containing the url")
    p.add_argument("--outdir", default="images", help="output folder")
    p.add_argument("--concurrency", type=int, default=8, help="parallel downloads")
    p.add_argument("--retries", type=int, default=2, help="retry attempts per image")
    p.add_argument("--timeout", type=int, default=20, help="per-image timeout in seconds")
    p.add_argument("--prefix", default="img", help="filename prefix")
    p.add_argument("--meta", default="download_meta.csv", help="metadata CSV output file")
    return p.parse_args()

async def main():
    args = parse_args()
    outdir = Path(args.outdir)
    urls = []
    if args.mode == "urls":
        inp = Path(args.input)
        if not inp.exists():
            print(f"Input file not found: {inp}", file=sys.stderr)
            sys.exit(2)
        if inp.suffix.lower() in [".txt", ".urls"]:
            urls = read_urls_from_txt(inp)
        elif inp.suffix.lower() in [".csv"]:
            urls = read_urls_from_csv(inp, colname=args.incol)
        else:
            # try txt by default
            urls = read_urls_from_txt(inp)
    else:  # crawl
        target = args.input
        # ensure it's a URL
        if not re.match(r"^https?://", target):
            print("For crawl mode, input must be a full URL (starting with http:// or https://)", file=sys.stderr)
            sys.exit(2)
        # fetch page
        async with aiohttp.ClientSession() as s:
            html = await fetch_html(s, target, timeout=15)
            if not html:
                print("Failed to fetch page or empty body.", file=sys.stderr)
                sys.exit(3)
            urls = extract_img_urls(target, html)
            print(f"Found {len(urls)} image URLs on {target}")

    if not urls:
        print("No URLs to download.", file=sys.stderr)
        sys.exit(0)

    # normalization: optionally filter out data: URIs and JS blobs
    urls = [u for u in urls if not u.startswith("data:") and re.match(r"^https?://", u)]
    print(f"Starting download of {len(urls)} images to {outdir}")

    results = await run_downloader(urls, outdir, concurrency=args.concurrency,
                                   retries=args.retries, timeout=args.timeout, prefix=args.prefix)

    # write metadata CSV
    meta_path = Path(args.meta)
    with open(meta_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["url", "file", "status", "size"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Done. Metadata written to {meta_path}. Successful: {sum(1 for r in results if r['status']=='ok')}, Failed: {sum(1 for r in results if r['status']!='ok')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
