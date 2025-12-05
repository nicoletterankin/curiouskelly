# Email Setup: hello@curiouskelly.com
## Get it working in 1 hour

---

## OPTION 1: GOOGLE WORKSPACE (RECOMMENDED)
**Cost:** $6/month  
**Time:** 30 minutes  
**Best for:** Professional, reliable, easy

### Steps:

1. **Go to:** https://workspace.google.com
2. **Click:** "Get Started"
3. **Enter business name:** Curious Kelly
4. **Number of employees:** 1
5. **Location:** United States (or your country)
6. **Enter domain:** curiouskelly.com
7. **Create admin account:**
   - Email: admin@curiouskelly.com
   - Password: [strong password]

8. **Verify domain ownership:**
   - Google gives you a TXT record
   - Go to your domain registrar (Namecheap/GoDaddy/Cloudflare)
   - Add TXT record to DNS
   - Wait 5-15 minutes
   - Click "Verify" in Google

9. **Set up MX records:**
   - Google provides MX records
   - Add to your DNS:
   ```
   Priority 1: ASPMX.L.GOOGLE.COM
   Priority 5: ALT1.ASPMX.L.GOOGLE.COM
   Priority 5: ALT2.ASPMX.L.GOOGLE.COM
   Priority 10: ALT3.ASPMX.L.GOOGLE.COM
   Priority 10: ALT4.ASPMX.L.GOOGLE.COM
   ```
   - Save changes

10. **Create hello@ email:**
    - Workspace admin → Users → Add user
    - Email: hello@curiouskelly.com
    - Name: Curious Kelly Support
    - Password: [strong password]

11. **Test it:**
    - Send email from hello@curiouskelly.com
    - Send email TO hello@curiouskelly.com
    - Both should work

✅ **DONE!** You have hello@curiouskelly.com

---

## OPTION 2: ZOHO MAIL (FREE)
**Cost:** Free (up to 5 users)  
**Time:** 30 minutes  
**Best for:** Budget-conscious

### Steps:

1. **Go to:** https://www.zoho.com/mail/
2. **Click:** "Sign Up Now - Free"
3. **Enter domain:** curiouskelly.com
4. **Create admin account**
5. **Verify domain:**
   - Zoho gives you TXT or CNAME record
   - Add to DNS
   - Verify
6. **Configure MX records:**
   ```
   Priority 10: mx.zoho.com
   Priority 20: mx2.zoho.com
   ```
7. **Create hello@ email**
8. **Test it**

✅ **DONE!**

---

## OPTION 3: CLOUDFLARE EMAIL ROUTING (FREE)
**Cost:** Free  
**Time:** 10 minutes  
**Best for:** Quick setup, forwards to your personal email

### Steps:

1. **Go to:** https://dash.cloudflare.com
2. **Select:** curiouskelly.com domain
3. **Click:** "Email" → "Email Routing"
4. **Click:** "Get Started"
5. **Add destination:**
   - Your personal email (Gmail, etc.)
   - Verify it (click link in email)
6. **Add route:**
   - Custom address: hello@curiouskelly.com
   - Destination: Your personal email
7. **Save**

**What this does:**
- Emails sent TO hello@curiouskelly.com forward to your inbox
- You can reply FROM hello@curiouskelly.com (with setup)

**Limitation:** Forwarding only, not full mailbox

✅ **DONE!** (Quick and dirty solution)

---

## OPTION 4: SENDGRID (For Automated Emails Only)
**Cost:** Free (100 emails/day), $15/mo (40k emails/month)  
**Time:** 20 minutes  
**Best for:** Sending automated emails (not receiving)

### Steps:

1. **Go to:** https://sendgrid.com
2. **Sign up:** Free account
3. **Verify email**
4. **Go to:** Settings → Sender Authentication
5. **Authenticate domain:** curiouskelly.com
6. **Add DNS records** (SendGrid provides)
7. **Create API key:**
   - Settings → API Keys → Create
   - Name: "Curious Kelly Backend"
   - Permissions: Full Access
   - Copy API key
8. **Add to backend .env:**
   ```
   SENDGRID_API_KEY=SG.xxxxxxxxxxxxx
   SENDGRID_FROM_EMAIL=hello@curiouskelly.com
   ```

**What this does:**
- Send emails FROM hello@curiouskelly.com
- Gift certificates, welcome emails, etc.

**Limitation:** Can't receive emails, only send

✅ **DONE!** Automated sending works

---

## RECOMMENDED SETUP

**Use both:**
1. **Google Workspace OR Zoho** (receiving emails, customer support)
2. **SendGrid** (automated emails, high volume)

**Why both:**
- Google/Zoho: Manual replies, customer support
- SendGrid: Automated gift emails, newsletters, transactional

**Cost:** $6/month (Google) + Free (SendGrid) = $6/month total

---

## TESTING YOUR EMAIL

### Test 1: Can you SEND?
```
1. Log into hello@curiouskelly.com
2. Compose email to your personal email
3. Send
4. Check personal inbox
5. ✅ Should arrive in 1-2 minutes
```

### Test 2: Can you RECEIVE?
```
1. Send email FROM your personal email
2. TO hello@curiouskelly.com
3. Check hello@ inbox
4. ✅ Should arrive in 1-2 minutes
```

### Test 3: Can you REPLY?
```
1. Reply to the test email
2. Check personal inbox
3. ✅ Should show "From: hello@curiouskelly.com"
```

### Test 4: Does automation work? (SendGrid)
```
1. Go to backend: curious-kellly/backend/
2. Create test script:
   const sgMail = require('@sendgrid/mail');
   sgMail.setApiKey(process.env.SENDGRID_API_KEY);
   
   sgMail.send({
     to: 'your-personal@email.com',
     from: 'hello@curiouskelly.com',
     subject: 'Test Email',
     text: 'If you see this, SendGrid works!'
   });
   
3. Run: node test-email.js
4. Check personal inbox
5. ✅ Should arrive in 1-2 minutes
```

---

## TROUBLESHOOTING

### "Email not arriving"
→ Check spam folder  
→ Wait 15 minutes (DNS propagation)  
→ Verify MX records correct in DNS

### "Can't verify domain"
→ DNS changes take 5-60 minutes  
→ Check TXT record copy-pasted exactly  
→ Remove any trailing spaces or dots

### "SendGrid blocked my account"
→ Common for new accounts  
→ Contact support: "I'm launching an educational product"  
→ Usually resolved in 24 hours

### "Sending works but not receiving"
→ MX records not set correctly  
→ Check DNS with: `dig MX curiouskelly.com`  
→ Should show Google or Zoho MX records

---

## DNS RECORDS SUMMARY

**Add these to curiouskelly.com DNS:**

### For Google Workspace:
```
Type: MX, Priority: 1, Value: ASPMX.L.GOOGLE.COM
Type: MX, Priority: 5, Value: ALT1.ASPMX.L.GOOGLE.COM
Type: MX, Priority: 5, Value: ALT2.ASPMX.L.GOOGLE.COM
Type: MX, Priority: 10, Value: ALT3.ASPMX.L.GOOGLE.COM
Type: MX, Priority: 10, Value: ALT4.ASPMX.L.GOOGLE.COM
```

### For SendGrid:
```
Type: CNAME, Name: em1234, Value: u1234567.wl.sendgrid.net
Type: CNAME, Name: s1._domainkey, Value: s1.domainkey.u1234567.wl.sendgrid.net
Type: CNAME, Name: s2._domainkey, Value: s2.domainkey.u1234567.wl.sendgrid.net
```
(SendGrid gives you exact values)

---

## SECURITY BEST PRACTICES

### Enable 2FA:
- [ ] Google Workspace: Enable 2-step verification
- [ ] Zoho: Enable 2-factor auth
- [ ] SendGrid: Enable 2FA in account settings

### Strong Password:
- [ ] 16+ characters
- [ ] Mix of upper, lower, numbers, symbols
- [ ] Unique (not used elsewhere)
- [ ] Store in password manager

### SPF/DKIM/DMARC:
- [ ] Add SPF record: `v=spf1 include:_spf.google.com ~all`
- [ ] Enable DKIM (Google/Zoho auto-configure)
- [ ] Add DMARC: `v=DMARC1; p=quarantine; rua=mailto:hello@curiouskelly.com`

---

## CONNECT TO BACKEND

Once email works, update backend:

**File:** `curious-kellly/backend/.env`

```bash
# Email Configuration
SENDGRID_API_KEY=SG.your_key_here
SENDGRID_FROM_EMAIL=hello@curiouskelly.com
SENDGRID_FROM_NAME=Curious Kelly

# Customer Support Email
SUPPORT_EMAIL=hello@curiouskelly.com
```

**Test backend email:**
```bash
cd curious-kellly/backend
npm run test:email
```

---

## EMAIL TEMPLATES READY

**Location:** `EMAIL_TEMPLATES_CHRISTMAS.md`

**Templates available:**
- Welcome email (after purchase)
- Gift certificate (Christmas morning)
- Daily lesson reminder
- Streak celebration
- Re-engagement
- And 9 more...

**Next step:** Import these into SendGrid templates

---

## TIMELINE

### Option 1 (Google Workspace):
- Setup: 30 minutes
- DNS propagation: 15 minutes
- Testing: 5 minutes
- **Total: ~1 hour**

### Option 3 (Cloudflare routing):
- Setup: 5 minutes
- DNS propagation: 5 minutes
- Testing: 2 minutes
- **Total: ~15 minutes**

---

## MY RECOMMENDATION

**For TODAY:**
Use Option 3 (Cloudflare Email Routing)
- Takes 15 minutes
- Free
- Forwards to your inbox
- UNBLOCKS SOCIAL ACCOUNT CREATION

**For LAUNCH:**
Upgrade to Option 1 (Google Workspace)
- Professional
- Full mailbox
- Better for customer support
- Do this next week

**For AUTOMATION:**
Set up Option 4 (SendGrid)
- Required for gift emails
- Set up next week
- Test before launch

---

## NEXT STEPS AFTER EMAIL WORKS

1. **Create social media accounts** (use hello@curiouskelly.com)
2. **Set up customer support** (replies go to hello@)
3. **Import email templates** to SendGrid
4. **Test gift purchase flow** (emails send correctly?)

---

**NOW GO SET UP THAT EMAIL! ⏱️ 15 minutes starting NOW!**














