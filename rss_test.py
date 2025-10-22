#!/usr/bin/env python3
from rss_parser import extract_article_fulltext

URL = "https://seekingalpha.com/news/4501274-cathie-wood-purchases-the-dip-grabs-over-500k-shares-of-the-sliding-draftkings?utm_source=feed_news_all&utm_medium=referral&feed_item_type=news"    

res = extract_article_fulltext(URL)

print("="*80)
print("FINAL URL:", res.get("final_url"))
print("TITLE    :", (res.get("title") or "").strip())
text = (res.get("text") or "").strip()
print("CHARS    :", len(text))
print("="*80)
print(text if text else "(no body extracted)")
print()