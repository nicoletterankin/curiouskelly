# Multi-Account Imagen Generation Setup

## Goal
Bypass per-project rate limits by using multiple Google Cloud projects, each with its own API key and quota.

---

## Quick Setup (per account)

### 1. Create New Project
```
https://console.cloud.google.com/projectcreate
```
- Name: `curious-kelly-gen-1`, `curious-kelly-gen-2`, etc.

### 2. Enable Generative Language API
```
https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
```
- Select your new project
- Click "Enable"

### 3. Link Billing Account
```
https://console.cloud.google.com/billing/linkedaccount
```
- Select your new project
- Link to your existing billing account (same credit card works for all projects)

### 4. Create API Key
```
https://console.cloud.google.com/apis/credentials
```
- Select your new project
- Click "Create Credentials" → "API Key"
- Copy the key

### 5. Add to .env.local
```bash
GOOGLE_API_KEY_1=AIza...original...
GOOGLE_API_KEY_2=AIza...new1...
GOOGLE_API_KEY_3=AIza...new2...
GOOGLE_API_KEY_4=AIza...new3...
```

---

## Quota per Project (Paid Tier 1)

| Model | Daily Limit | Cost |
|-------|-------------|------|
| imagen-4.0-ultra-generate-001 | 30 | $0.06 |
| imagen-4.0-generate-001 | 30 | $0.04 |
| imagen-4.0-fast-generate-001 | 30 | $0.02 |
| **Total per project** | **90** | ~$4.20 |

---

## Coverage Calculator

| # of Projects | Images/Day | Days to Cover 365×7 |
|---------------|------------|---------------------|
| 1 | 90 | 28 days |
| 2 | 180 | 14 days |
| 3 | 270 | 10 days |
| 5 | 450 | 6 days |
| 10 | 900 | 3 days |

---

## Recommended: Create 5 Projects

With 5 projects:
- 450 images/day
- Full year coverage in ~6 days
- Cost: ~$150 total

With 10 projects:
- 900 images/day  
- Full year coverage in **3 days**
- Cost: ~$150 total (same, just faster)

---

## Script Usage

After adding keys to .env.local:

```bash
npx tsx scripts/multi-key-generator.ts --range=1-365
```

The script will automatically rotate through available API keys.
