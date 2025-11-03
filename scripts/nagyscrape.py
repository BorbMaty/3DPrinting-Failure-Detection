#!/usr/bin/env python3
# nagyscrape_v2.py – Google Images + Bing fallback, cookie-fal kezeléssel
# by Selene ✨

import argparse, hashlib, io, os, time, pathlib, json, re, random
from typing import Set, Tuple, List
from urllib.parse import quote_plus
import requests
from PIL import Image, UnidentifiedImageError

# ---- Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")

def sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1(); h.update(data); return h.hexdigest()

def robust_get(url: str, timeout: int = 20) -> requests.Response:
    headers = {
        "User-Agent": UA,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

def save_image_bytes(out_dir: pathlib.Path, img_bytes: bytes, min_size: int, used_hashes: Set[str]) -> Tuple[bool, str]:
    h = sha1_bytes(img_bytes)
    if h in used_hashes:
        return False, "duplicate"
    try:
        im = Image.open(io.BytesIO(img_bytes)); im.verify()
        im = Image.open(io.BytesIO(img_bytes))
        if min(im.size) < min_size:
            return False, f"too_small_{im.size}"
        ext = { "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp" }.get(im.format, ".jpg")
    except UnidentifiedImageError:
        return False, "unidentified"
    except Exception as e:
        return False, f"verify_fail_{e.__class__.__name__}"
    (out_dir / f"{h}{ext}").write_bytes(img_bytes)
    used_hashes.add(h)
    return True, f"{h}{ext}"

def setup_driver(headless: bool) -> webdriver.Chrome:
    opts = Options()
    if headless:
        # headless=new néha blokkol – próbáld ki nélküle is, ha gond van
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(f"user-agent={UA}")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    svc = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=svc, options=opts)
    driver.set_page_load_timeout(45)
    return driver

def google_images_url(query: str) -> str:
    # udm=2: “image viewer v2” – stabilabb lett a markup
    q = quote_plus(query)
    return f"https://www.google.com/search?tbm=isch&udm=2&hl=en&gl=us&q={q}"

def try_accept_cookies(driver):
    # több régiós variáció
    for sel in [
        "button#L2AGLb",                         # EU
        "button[aria-label='Accept all']",       # néha ez
        "button[aria-label='I agree']",
        "button[role='button'] div:contains('I agree')", # kevésbé stabil
    ]:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            for b in btns:
                if b.is_displayed():
                    b.click(); time.sleep(1.0); return
        except Exception:
            pass
    # későbbi modal
    try:
        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button#L2AGLb"))
        ).click()
        time.sleep(1.0)
    except Exception:
        pass

THUMB_SELECTORS = [
    "img.rg_i", "img.YQ4gaf", "img.Q4LuWd", "img.rQINMb", "img[jsname='Q4LuWd']"
]

def wait_for_thumbs(driver, timeout=10):
    for sel in THUMB_SELECTORS:
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            return sel
        except TimeoutException:
            continue
    return None

def extract_fullsize_candidates(driver) -> List[str]:
    # nagy kép a jobb panelben: img.n3VNCb, img.sFlh5c – variálódik
    cands, seen = [], set()
    for sel in ["img.n3VNCb", "img.sFlh5c", "img[alt][src^='http']"]:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            src = el.get_attribute("src") or ""
            if src.startswith("http") and src not in seen:
                seen.add(src); cands.append(src)
    return cands

def google_scrape_class(driver, query: str, out_dir: pathlib.Path, target: int, min_size: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    used_hashes: Set[str] = set(p.stem for p in out_dir.glob("*.*"))
    downloaded = len(used_hashes)
    driver.get(google_images_url(query)); time.sleep(2.0)
    try_accept_cookies(driver)
    sel = wait_for_thumbs(driver, timeout=10)
    if not sel:
        print(f"[WARN] No thumbnails detected on Google for '{query}' (consent/wall?).")
        return downloaded

    body = driver.find_element(By.TAG_NAME, "body")
    seen = set()
    scrolls = 0

    while downloaded < target and scrolls < 180:
        thumbs = driver.find_elements(By.CSS_SELECTOR, sel)
        for t in thumbs:
            tid = id(t)
            if tid in seen: continue
            seen.add(tid)
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", t)
                time.sleep(0.15)
                t.click(); time.sleep(0.35)
            except WebDriverException:
                continue

            # jelölt nagy képek
            for src in extract_fullsize_candidates(driver):
                try:
                    resp = robust_get(src, timeout=25)
                    if resp.status_code != 200: continue
                    ok, _ = save_image_bytes(out_dir, resp.content, min_size, used_hashes)
                    if ok:
                        downloaded += 1
                        print(f"[Google:{query}] {downloaded}/{target}")
                        if downloaded >= target: break
                except requests.RequestException:
                    pass
            if downloaded >= target: break

        # lap aljára görgetés + több találat
        body.send_keys(Keys.END); time.sleep(0.6)
        # Show more
        for more_sel in ["input.mye4qd", "span.Q2MMlc", "button[jsname='j6LnYe']"]:
            try:
                for b in driver.find_elements(By.CSS_SELECTOR, more_sel):
                    if b.is_displayed():
                        driver.execute_script("arguments[0].click();", b)
                        time.sleep(1.0)
            except Exception:
                pass
        scrolls += 1

    print(f"[DONE Google] '{query}' -> +{downloaded - (len(used_hashes))} (total in dir now: {len(list(out_dir.glob('*.*')))} )")
    return downloaded

# --------- Bing fallback (requests + regex/JSON scrape, nincs Selenium)
def bing_image_urls(query: str, max_urls: int = 200) -> List[str]:
    # Egyszerű scraper: a HTML-ben gyakran vannak "murl":"<url>" mezők (media url)
    # Több lapot is kérünk &first param-mal.
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9",
    }
    urls, seen = [], set()
    step = 50
    for start in range(0, max_urls*2, step):
        u = f"https://www.bing.com/images/async?q={quote_plus(query)}&first={start}&count={step}&qft=+filterui:imagesize-large&form=IBASEP"
        try:
            r = requests.get(u, headers=headers, timeout=20)
            if r.status_code != 200: continue
            # Keressünk murl JSON mezőket
            for m in re.finditer(r'"murl":"(https?://[^"]+)"', r.text):
                url = m.group(1).encode('utf-8').decode('unicode_escape')
                if url not in seen:
                    seen.add(url); urls.append(url)
            if len(urls) >= max_urls: break
        except requests.RequestException:
            continue
        time.sleep(0.6 + random.random()*0.4)
    return urls[:max_urls]

def bing_download(out_dir: pathlib.Path, query: str, need: int, min_size: int) -> int:
    used_hashes: Set[str] = set(p.stem for p in out_dir.glob("*.*"))
    downloaded0 = len(used_hashes)
    urls = bing_image_urls(query, max_urls=max(need*3, 200))
    for u in urls:
        if len(used_hashes) - downloaded0 >= need: break
        try:
            resp = robust_get(u, timeout=25)
            if resp.status_code != 200: continue
            ok, _ = save_image_bytes(out_dir, resp.content, min_size, used_hashes)
            if ok:
                print(f"[Bing:{query}] {(len(used_hashes)-downloaded0)}/{need}")
        except requests.RequestException:
            continue
    return len(used_hashes) - downloaded0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dataset/images_raw"))
    ap.add_argument("--min-size", type=int, default=128)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--headless", action="store_true", help="Force headless Chrome")
    g.add_argument("--no-headless", action="store_true", help="Force visible Chrome")
    args = ap.parse_args()

    headless = True
    if args.no_headless: headless = False
    if args.headless: headless = True

    driver = setup_driver(headless=headless)
    try:
        for cls in args.classes:
            query = f"3D printing {cls.replace('_',' ')} issue defect problem"
            out_dir = args.outdir / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            before = len(list(out_dir.glob("*.*")))
            got_google = google_scrape_class(driver, query, out_dir, args.per_class, args.min_size)
            after_g = len(list(out_dir.glob("*.*")))
            have = after_g - before

            if have < args.per_class:
                need = args.per_class - have
                print(f"[INFO] Google gave {have}. Trying Bing fallback for '{cls}' (+{need}).")
                more = bing_download(out_dir, f"3D printing {cls.replace('_',' ')} failure", need, args.min_size)
                print(f"[DONE Bing] '{cls}' -> +{more} (total now: {len(list(out_dir.glob('*.*')))} )")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
