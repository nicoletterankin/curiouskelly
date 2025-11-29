# 🔐 Secrets Quick Reference Card
**Print this and keep it handy!**

---

## 🚀 Most Common Secrets (Copy-Paste Ready)

### Stripe (Payment Processing)
```
Dashboard: https://dashboard.stripe.com
→ Developers → API keys → Secret key
→ Products → [Product] → Pricing → Price ID
→ Developers → Webhooks → [Endpoint] → Signing secret
```

### Supabase (Database)
```
Dashboard: https://app.supabase.com
→ Project → Settings → API
  - Project URL → PUBLIC_SUPABASE_URL
  - anon public → PUBLIC_SUPABASE_ANON_KEY
  - service_role → SUPABASE_SERVICE_ROLE_KEY (SECRET!)
→ Settings → Database → Connection string → SUPABASE_DB_URL
```

### Vercel (Deployment)
```
Dashboard: https://vercel.com/dashboard
→ Project → Settings → Environment Variables
→ Account → Tokens → VERCEL_TOKEN (for CI/CD)
```

---

## 📍 Where to Add Secrets

### Local Development
```bash
# 1. Copy template
cp .env.example .env

# 2. Edit with your values
notepad .env  # Windows
nano .env     # Mac/Linux
```

### Production (Vercel)
```
https://vercel.com/dashboard
→ [Your Project]
→ Settings
→ Environment Variables
→ Add (for each variable)
```

### Production (GitHub Actions)
```
https://github.com/[org]/[repo]/settings/secrets/actions
→ New repository secret
→ Add each secret
```

---

## ⚠️ Critical Rules

1. ✅ **DO:** Use `.env` locally (gitignored)
2. ✅ **DO:** Use `.env.example` as template (committed)
3. ✅ **DO:** Add secrets to Vercel/GitHub for production
4. ❌ **DON'T:** Commit `.env` files
5. ❌ **DON'T:** Hardcode secrets in code
6. ❌ **DON'T:** Share secrets in chat/email

---

## 🆘 Lost a Secret?

- **Stripe:** Dashboard → Developers → API keys → Reveal key
- **Supabase:** Dashboard → Settings → API → Reset key (⚠️ revokes old one!)
- **Vercel:** Dashboard → Account → Tokens → Create new token

---

## 📚 Full Documentation

See `SECRETS_MASTER_REFERENCE.md` for complete details.





