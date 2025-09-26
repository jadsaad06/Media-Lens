import os
import feedparser
from urllib.parse import urlparse
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import text
from utils.db import SessionLocal
from utils.canonical import canonicalize_url
from utils.robots import robots_ok

load_dotenv()

DEFAULT_FEEDS = [u.strip() for u in os.getenv('RSS_FEEDS','').split(',') if u.strip()]

def parse_published(entry):
    # feedparser gives .published_parsed; fallback to None
    try:
        t = entry.published_parsed
        return datetime(*t[:6], tzinfo=timezone.utc)
    except Exception:
        return None

def normalize_entry(entry, source_url):
    url = entry.get('link')
    title = entry.get('title')
    summary = entry.get('summary') or entry.get('description')
    published_at = parse_published(entry)
    canonical = canonicalize_url(url) if url else url
    domain = urlparse(url).netloc if url else None
    return {
        'url': url,
        'canonical_url': canonical,
        'domain': domain,
        'source': source_url,
        'title': title,
        'published_at': published_at,
        'summary': summary,
        'text': None
    }

def upsert_article(sess, a):
    # simple content hash (url as unique); for Week 1 we just enforce URL unique
    sql = text("""
        INSERT INTO articles (url, canonical_url, domain, source, title, published_at, summary, text, hash, robots_ok, text_limited)
        VALUES (:url, :canonical_url, :domain, :source, :title, :published_at, :summary, :text, :hash, :robots_ok, :text_limited)
        ON CONFLICT (url) DO UPDATE SET
          canonical_url = EXCLUDED.canonical_url,
          domain = EXCLUDED.domain,
          source = EXCLUDED.source,
          title = EXCLUDED.title,
          published_at = COALESCE(EXCLUDED.published_at, articles.published_at),
          summary = COALESCE(EXCLUDED.summary, articles.summary)
        RETURNING id;
    """)
    return sess.execute(sql, a).scalar()

def main(feeds=None):
    feeds = feeds or DEFAULT_FEEDS
    if not feeds:
        print("No feeds configured. Set RSS_FEEDS or pass URLs via CLI.")
        return
    sess = SessionLocal()
    inserted = 0
    for f in feeds:
        d = feedparser.parse(f)
        for e in d.entries:
            norm = normalize_entry(e, f)
            if not norm['url']:
                continue
            ok = robots_ok(norm['url'])
            norm['robots_ok'] = bool(ok)
            norm['text_limited'] = not ok
            norm['hash'] = norm['canonical_url'] or norm['url']
            try:
                _id = upsert_article(sess, norm)
                inserted += 1 if _id else 0
            except Exception as ex:
                sess.rollback()
                print("Skip due to error:", ex)
            else:
                sess.commit()
    print(f"Upserted {inserted} article records.")
    sess.close()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
