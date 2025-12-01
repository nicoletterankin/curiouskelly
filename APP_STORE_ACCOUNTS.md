# 🔐 App Store Accounts & Credentials

**Purpose**: Centralized location for all app store account information  
**Security**: DO NOT commit credentials to git  
**Status**: Awaiting account details from user

---

## 📱 APPLE DEVELOPER ACCOUNT

### Account Information
```
Email: [NEEDED]
Team ID: [NEEDED]
Team Name: Lesson of the Day PBC
Organization Type: Public Benefit Corporation
```

### Access URLs
- **App Store Connect**: https://appstoreconnect.apple.com
- **Developer Portal**: https://developer.apple.com/account
- **Certificates**: https://developer.apple.com/account/resources/certificates/list

### Required Information
- [ ] Apple ID email
- [ ] Team ID (10-character identifier)
- [ ] App-specific password (for CI/CD)
- [ ] Distribution certificate
- [ ] Provisioning profiles

### App Information
```
App Name: Curious Kelly
Bundle ID: com.curiouskelly.mobile (iOS)
           com.curiouskelly.desktop (macOS)
SKU: CURIOUS-KELLY-001
Primary Language: English (US)
Category: Education
```

### Pricing
```
Price Tier: Free (with in-app purchases)
In-App Purchases:
  - Monthly: $9.99/month
  - Annual: $99/year
  - Lifetime: $499 one-time
```

---

## 🤖 GOOGLE PLAY DEVELOPER ACCOUNT

### Account Information
```
Email: [NEEDED]
Developer ID: [NEEDED]
Developer Name: Lesson of the Day PBC
```

### Access URLs
- **Play Console**: https://play.google.com/console
- **Developer Account**: https://play.google.com/apps/publish

### Required Information
- [ ] Google account email
- [ ] Developer account ID
- [ ] Payment merchant ID
- [ ] Service account JSON (for CI/CD)
- [ ] Signing key

### App Information
```
App Name: Curious Kelly
Package Name: com.curiouskelly.mobile
Content Rating: Everyone
Category: Education
```

### Pricing
```
Price: Free (with in-app purchases)
In-App Products:
  - monthly_subscription: $9.99/month
  - annual_subscription: $99/year
  - lifetime_access: $499 one-time
```

---

## 📺 ROKU DEVELOPER ACCOUNT

### Account Information
```
Email: [NEEDED]
Developer ID: [NEEDED]
Developer Name: Lesson of the Day PBC
```

### Access URLs
- **Developer Dashboard**: https://developer.roku.com
- **Channel Management**: https://developer.roku.com/apps

### Required Information
- [ ] Roku account email
- [ ] Developer ID
- [ ] Payment information
- [ ] Tax information

### Channel Information
```
Channel Name: Curious Kelly
Channel ID: [Will be assigned]
Category: Education & Learning
Rating: G (General Audiences)
Price: Free
```

---

## 🐙 GITHUB ORGANIZATION

### Account Information
```
Organization: curiouskelly
Repository: desktop-app
Owner: [NEEDED]
```

### Access URLs
- **Organization**: https://github.com/curiouskelly
- **Desktop App**: https://github.com/curiouskelly/desktop-app
- **Releases**: https://github.com/curiouskelly/desktop-app/releases

### Required Information
- [ ] GitHub organization owner
- [ ] Access tokens for CI/CD
- [ ] Repository settings
- [ ] Release permissions

### Release URLs (After First Release)
```
Windows: https://github.com/curiouskelly/desktop-app/releases/latest/download/Curious-Kelly-Setup.exe
macOS: https://github.com/curiouskelly/desktop-app/releases/latest/download/Curious-Kelly.dmg
Linux: https://github.com/curiouskelly/desktop-app/releases/latest/download/Curious-Kelly.AppImage
```

---

## 📧 EMAIL ACCOUNTS

### Primary Contact
```
Email: hello@curiouskelly.com
Purpose: All customer/community communications
Status: [VERIFY WORKING]
```

### Required Checks
- [ ] Email receives messages
- [ ] Auto-responder configured
- [ ] Forwarding rules set up
- [ ] Spam filters configured
- [ ] Response time < 24 hours

---

## 🔑 API KEYS & SECRETS

### Supabase
```
Project URL: [FROM CLOUDFLARE_R2_CREDENTIALS.md]
Anon Key: [FROM CLOUDFLARE_R2_CREDENTIALS.md]
Service Role Key: [FROM CLOUDFLARE_R2_CREDENTIALS.md]
```

### Stripe
```
Publishable Key: [NEEDED]
Secret Key: [NEEDED]
Webhook Secret: [NEEDED]
```

### ElevenLabs
```
API Key: [NEEDED]
Voice IDs:
  - Kelly: [NEEDED]
  - Kyle: [NEEDED]
```

---

## 🚀 DEPLOYMENT CREDENTIALS

### Vercel
```
Project: curiouskelly
Organization: lotd
Token: [NEEDED for CI/CD]
```

### Cloudflare
```
Account ID: [FROM CLOUDFLARE_R2_CREDENTIALS.md]
R2 Access Key: [FROM CLOUDFLARE_R2_CREDENTIALS.md]
R2 Secret Key: [FROM CLOUDFLARE_R2_CREDENTIALS.md]
```

---

## ✅ ACCOUNT SETUP CHECKLIST

### Apple Developer Account
- [ ] Account created and verified
- [ ] $99/year fee paid
- [ ] Team created (Lesson of the Day PBC)
- [ ] Tax forms submitted
- [ ] Banking information added
- [ ] Agreements accepted

### Google Play Developer Account
- [ ] Account created and verified
- [ ] $25 one-time fee paid
- [ ] Developer profile completed
- [ ] Tax information submitted
- [ ] Banking information added
- [ ] Content rating questionnaire completed

### Roku Developer Account
- [ ] Account created and verified
- [ ] Developer profile completed
- [ ] Tax information submitted
- [ ] Banking information added
- [ ] Channel properties configured

### GitHub Organization
- [ ] Organization created
- [ ] Repository created (desktop-app)
- [ ] README and documentation added
- [ ] Release workflow configured
- [ ] Access permissions set

### Email Account
- [ ] hello@curiouskelly.com configured
- [ ] Email tested (send/receive)
- [ ] Auto-responder set up
- [ ] Forwarding configured
- [ ] Monitoring enabled

---

## 📞 SUPPORT CONTACTS

### Apple Developer Support
- **Phone**: 1-800-633-2152 (US)
- **Email**: https://developer.apple.com/contact/
- **Hours**: 24/7 for urgent issues

### Google Play Support
- **Help Center**: https://support.google.com/googleplay/android-developer
- **Community**: https://www.reddit.com/r/androiddev
- **Response Time**: 1-3 business days

### Roku Developer Support
- **Email**: developer@roku.com
- **Forum**: https://community.roku.com/t5/Developers/ct-p/channel-developers
- **Response Time**: 2-5 business days

---

## 🎯 NEXT STEPS

### Immediate (Today)
1. Gather all account credentials
2. Verify email is working
3. Test access to all platforms
4. Document any missing information

### This Week (Dec 1-7)
1. Complete all account setups
2. Submit tax/banking information
3. Create app listings
4. Prepare all assets

### Before Submission
1. Verify all accounts active
2. Test all access credentials
3. Confirm payment methods
4. Review all agreements

---

## 🔒 SECURITY NOTES

### Best Practices
- ✅ Use strong, unique passwords
- ✅ Enable 2FA on all accounts
- ✅ Store credentials in secure password manager
- ✅ Never commit credentials to git
- ✅ Rotate API keys regularly
- ✅ Monitor for unauthorized access

### Access Control
- Limit access to essential personnel only
- Use service accounts for CI/CD
- Audit access logs regularly
- Revoke unused credentials immediately

---

**Status**: Awaiting account credentials from user  
**Action Required**: Fill in [NEEDED] fields above  
**Priority**: HIGH - Required for December 17 launch



