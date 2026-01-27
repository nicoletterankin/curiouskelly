# CURSOR: DEPLOY LESSON OF THE DAY SUBDOMAINS

## YOUR MISSION

Deploy two Cloudflare Pages sites:
1. **invest.thedailylesson.com** — Investor video landing page
2. **dallas.thedailylesson.com** — Dallas Short's command center

Do NOT touch thedailylesson.com or curiouskelly.com.

## CREDENTIALS

Account ID: `47ebb2a1adc311cb106acc89720e352c`

API Token: Ask user to get from https://dash.cloudflare.com/profile/api-tokens
- Template: "Edit Cloudflare Workers"
- Add: Zone > DNS > Edit
- Zone: thedailylesson.com

## EXECUTION SEQUENCE

### 1. Environment Setup
```bash
export CLOUDFLARE_ACCOUNT_ID="47ebb2a1adc311cb106acc89720e352c"
export CLOUDFLARE_API_TOKEN="<USER_PROVIDES>"
npm install -g wrangler
```

### 2. Get the deploy zips
Download from Claude outputs or user's local files:
- invest.zip → extract to ./invest/
- dallas.zip → extract to ./dallas/

### 3. Deploy
```bash
wrangler pages deploy ./invest --project-name=invest-dailylesson
wrangler pages deploy ./dallas --project-name=dallas-dailylesson
```

### 4. Custom Domains
```bash
curl -X POST "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/invest-dailylesson/domains" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"invest.thedailylesson.com"}'

curl -X POST "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/dallas-dailylesson/domains" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"dallas.thedailylesson.com"}'
```

### 5. Verify
```bash
curl -I https://invest.thedailylesson.com
curl -I https://dallas.thedailylesson.com
```

## SUCCESS OUTPUT

```
✅ invest.thedailylesson.com — LIVE
✅ dallas.thedailylesson.com — LIVE
```

## IF ERRORS

- "Authentication failed" → Token wrong or missing
- "Project exists" → Skip project create, just deploy
- "Domain in use" → Check if already configured

## GO
