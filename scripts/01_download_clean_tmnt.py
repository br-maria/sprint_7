import os
import time
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from slugify import slugify  # pip install python-slugify

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"
RAW_DIR.mkdir(exist_ok=True, parents=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (RAG-Study-TMNT; +https://example.com)"
}

# Список ссылок на страницы TMNT-вики
URLS_FILE = BASE_DIR / "tmnt_urls.json"

def load_urls():
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Пример для Fandom / MediaWiki
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        content = soup.body

    # Удаляем скрипты, стили, навигацию
    for tag in content(["script", "style", "noscript"]):
        tag.decompose()

    text = content.get_text(separator="\n")
    # Убираем лишние пустые строки
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)

def download_page(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text

from requests.exceptions import HTTPError

def main():
    urls = load_urls()
    print(f"Loaded {len(urls)} URLs")

    for url in urls:
        print(f"Processing: {url}")
        try:
            html = download_page(url)
        except HTTPError as e:
            print(f"❌ Skipping {url}: {e}")
            continue

        text = clean_html_to_text(html)

        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title_raw = m.group(1)
            title_raw = title_raw.split("|")[0].strip()
        else:
            title_raw = url.rsplit("/", 1)[-1]

        slug = slugify(title_raw)
        out_path = RAW_DIR / f"{slug}.md"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {title_raw}\n\n")
            f.write(text)

        print(f"Saved: {out_path}")
        time.sleep(1.0)

if __name__ == "__main__":
    main()
