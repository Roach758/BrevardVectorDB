"""
Downloader for Brevard Public Schools agendas, minutes, and attachments
from the district's Agenda Plus portal (agenda.brevardschools.org).
Fills the gap for August-December 2024 and any missing 2025/2026 meetings.
"""

import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

PORTAL_URL = "https://agenda.brevardschools.org/publishing/ap-agendas.html"
BASE_PUBLISHING = "https://agenda.brevardschools.org/publishing/"
BASE_URL = "https://agenda.brevardschools.org"
AGENDAS_DIR = Path(__file__).resolve().parent / "agendas"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def sanitize_filename(name: str) -> str:
    """Sanitize string for valid directory/file names on Windows."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def fetch_bytes(url: str, max_retries: int = 3, timeout: int = 20) -> bytes | None:
    """Fetch raw bytes from a URL with retries."""
    # Ensure URL is properly encoded
    parsed = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parsed.path)
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, encoded_path, parsed.query, parsed.fragment))
    
    req = urllib.request.Request(safe_url, headers=HEADERS)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def fetch_portal_meetings() -> list[dict]:
    """Scrape all table rows from the Agenda Plus portal."""
    print("Connecting to Brevard Public Schools Agenda portal...")
    raw = fetch_bytes(PORTAL_URL)
    if not raw:
        raise RuntimeError("Failed to fetch portal homepage")
        
    soup = BeautifulSoup(raw.decode("utf-8", errors="ignore"), "html.parser")
    meetings = []
    
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
            
        date_raw = tds[0].get_text(strip=True)
        title_raw = tds[1].get_text(strip=True)
        
        m = re.search(r'([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})', date_raw)
        if not m:
            continue
            
        month_name, day, year_str = m.groups()
        year = int(year_str)
        try:
            dt = datetime.strptime(f"{month_name} {int(day):02d} {year}", "%B %d %Y")
            iso_date = dt.strftime("%Y-%m-%d")
        except Exception:
            iso_date = f"{year}-00-00"
            
        # Target criteria:
        # August-December 2024, or 2025, or 2026
        is_target = False
        if year == 2024 and month_name.lower() in ["august", "september", "october", "november", "december"]:
            is_target = True
        elif year in [2025, 2026]:
            is_target = True
            
        if not is_target:
            continue
            
        links = {}
        for a in tr.find_all("a", href=True):
            href = a["href"].strip()
            l_text = a.get_text(strip=True).lower()
            if "agenda" in l_text or "agenda.html" in href:
                links["agenda"] = urllib.parse.urljoin(BASE_PUBLISHING, href)
            elif "minutes" in l_text or "minutes.html" in href:
                links["minutes"] = urllib.parse.urljoin(BASE_PUBLISHING, href)
                
        if links:
            meetings.append({
                "iso_date": iso_date,
                "date_str": date_raw,
                "title": title_raw,
                "links": links,
            })
            
    return meetings


def extract_attachment_links(html_content: str, source_url: str) -> list[tuple[str, str]]:
    """Extract all attachment URLs (PDFs, docx, etc.) from an agenda or minutes HTML page."""
    soup = BeautifulSoup(html_content, "html.parser")
    attachments = []
    seen_urls = set()
    
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
            
        full_url = urllib.parse.urljoin(source_url, href)
        low_url = full_url.lower()
        
        # We want PDFs and files from /content/files/
        if (
            low_url.endswith(".pdf")
            or low_url.endswith(".docx")
            or low_url.endswith(".xlsx")
            or "/content/files/" in low_url
        ):
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                # Determine sensible filename
                url_filename = urllib.parse.unquote(full_url.split("/")[-1].split("?")[0])
                if not url_filename or "." not in url_filename:
                    link_text = sanitize_filename(a.get_text(strip=True))
                    url_filename = f"{link_text}.pdf" if link_text else "attachment.pdf"
                else:
                    url_filename = sanitize_filename(url_filename)
                attachments.append((url_filename, full_url))
                
    return attachments


def process_meeting(meeting: dict) -> dict:
    """Download HTML files and all attachments for a single meeting."""
    iso_date = meeting["iso_date"]
    clean_title = sanitize_filename(meeting["title"])
    folder_name = f"{iso_date} - {clean_title}"
    meeting_dir = AGENDAS_DIR / folder_name
    meeting_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "title": clean_title,
        "date": iso_date,
        "html_downloaded": 0,
        "pdf_downloaded": 0,
        "pdf_skipped": 0,
        "failed": 0,
    }
    
    attachments_to_download = []
    
    # 1. Download Agenda HTML
    agenda_url = meeting["links"].get("agenda")
    if agenda_url:
        agenda_path = meeting_dir / "agenda.html"
        if not agenda_path.exists():
            html_bytes = fetch_bytes(agenda_url)
            if html_bytes:
                agenda_path.write_bytes(html_bytes)
                stats["html_downloaded"] += 1
                attachments_to_download.extend(extract_attachment_links(html_bytes.decode("utf-8", errors="ignore"), agenda_url))
        else:
            # Parse existing agenda for attachments
            html_text = agenda_path.read_text(encoding="utf-8", errors="ignore")
            attachments_to_download.extend(extract_attachment_links(html_text, agenda_url))
            
    # 2. Download Minutes HTML
    minutes_url = meeting["links"].get("minutes")
    if minutes_url:
        minutes_path = meeting_dir / "minutes.html"
        if not minutes_path.exists():
            min_bytes = fetch_bytes(minutes_url)
            if min_bytes:
                minutes_path.write_bytes(min_bytes)
                stats["html_downloaded"] += 1
                attachments_to_download.extend(extract_attachment_links(min_bytes.decode("utf-8", errors="ignore"), minutes_url))
        else:
            min_text = minutes_path.read_text(encoding="utf-8", errors="ignore")
            attachments_to_download.extend(extract_attachment_links(min_text, minutes_url))
            
    # Deduplicate attachments
    unique_attachments = {}
    for fname, u in attachments_to_download:
        if fname not in unique_attachments:
            unique_attachments[fname] = u
            
    # 3. Download Attachments
    for fname, attach_url in unique_attachments.items():
        dest_file = meeting_dir / fname
        if dest_file.exists() and dest_file.stat().st_size > 0:
            stats["pdf_skipped"] += 1
            continue
            
        file_bytes = fetch_bytes(attach_url)
        if file_bytes:
            dest_file.write_bytes(file_bytes)
            stats["pdf_downloaded"] += 1
        else:
            stats["failed"] += 1
            
    return stats


def main():
    print("=" * 70)
    print("Brevard Public Schools - Agenda & Minutes Archive Downloader")
    print(f"Target Directory: {AGENDAS_DIR}")
    print("=" * 70)
    
    AGENDAS_DIR.mkdir(parents=True, exist_ok=True)
    meetings = fetch_portal_meetings()
    print(f"Identified {len(meetings)} meetings in target scope (Aug-Dec 2024, 2025, 2026).\n")
    
    # Process with thread pool for high speed
    total_html = 0
    total_pdf = 0
    total_skipped = 0
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_meeting, m): m for m in meetings}
        for future in as_completed(futures):
            m = futures[future]
            try:
                res = future.result()
                total_html += res["html_downloaded"]
                total_pdf += res["pdf_downloaded"]
                total_skipped += res["pdf_skipped"]
                print(f"[{res['date']}] {res['title'][:48]:<48} | HTML: +{res['html_downloaded']} | PDF: +{res['pdf_downloaded']} (Existing: {res['pdf_skipped']})")
            except Exception as e:
                print(f"[{m['iso_date']}] Error processing {m['title']}: {e}")
                
    print("\n" + "=" * 70)
    print("Download Summary:")
    print(f"Total HTML files downloaded: {total_html}")
    print(f"Total PDF/Doc attachments downloaded: {total_pdf}")
    print(f"Already existing attachments skipped: {total_skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
