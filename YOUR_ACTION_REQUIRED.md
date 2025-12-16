# Action Required: Google Search Console

## Why this matters
Google indexing quality depends on Search Console verification + sitemap submission. This is the single highest-leverage “discoverability foundation” task.

## What’s already live (confirmed)
- **`/sitemap.xml`**: live and returning **HTTP 200**
- **`/robots.txt`**: live and references:
  - `https://www.curiouskelly.com/sitemap.xml`
  - `https://www.curiouskelly.com/sitemap-lessons.xml` (also live, **HTTP 200**)

## Steps (HTML file verification — recommended)
1. Go to Google Search Console: `https://search.google.com/search-console`
2. Click **Add property**
3. Enter: **`https://www.curiouskelly.com`**
4. Choose verification method: **HTML file**
5. Download the verification file (it will be named like `googleXXXXXXXXXXXX.html`)
6. Add the downloaded file to the repo at: **`public/<that-file>.html`**
7. Deploy (push to `main`)
8. Return to Search Console and click **Verify**

## Submit sitemaps (2 minutes)
1. In Search Console: open **Sitemaps**
2. Submit:
   - `https://www.curiouskelly.com/sitemap.xml`
   - `https://www.curiouskelly.com/sitemap-lessons.xml`

## Status checklist
- [ ] Property added
- [ ] Ownership verified
- [ ] `sitemap.xml` submitted
- [ ] `sitemap-lessons.xml` submitted

## If verification fails
- Make sure the verification HTML file is reachable at:
  - `https://www.curiouskelly.com/googleXXXXXXXXXXXX.html`
- Make sure you uploaded the file to **`public/`** (the deployed output directory).



























