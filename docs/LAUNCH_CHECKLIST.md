# December 17 Launch Checklist

## Day Before (Dec 16)

### Technical
- [ ] Run health check: `curl https://curiouskelly.com/api/health`
- [ ] Check all cron jobs are registered in Vercel dashboard
- [ ] Verify CRON_SECRET and SUPABASE_WEBHOOK_SECRET in Vercel
- [ ] Test signup flow end-to-end (use incognito)
- [ ] Test daily email by manual trigger
- [ ] Check database backup status in Supabase dashboard
- [ ] Review error logs in Vercel (last 24h)

### Content
- [ ] Verify Day 351 lesson exists (Dec 17 = day 351)
- [ ] Confirm all 365 lessons have emoji and title
- [ ] Test lesson page: `curiouskelly.com/day/351`

### Legal
- [ ] Privacy policy accessible: `curiouskelly.com/privacy.html`
- [ ] Terms of service accessible: `curiouskelly.com/terms.html`
- [ ] Age gate working on signup (13+)
- [ ] Unsubscribe link works in email footer

### Monitoring
- [ ] Status page live: `curiouskelly.com/status.html`
- [ ] External monitoring set up (BetterUptime/UptimeRobot)
- [ ] hello@curiouskelly.com receiving test emails

---

## Launch Morning (Dec 17)

### 6:00 AM EST
- [ ] Check status page - all green
- [ ] Check health endpoint
- [ ] Verify cron ran at 7am EST (or trigger manually)

### 7:00 AM EST (Emails Send)
- [ ] Monitor Resend dashboard for sends
- [ ] Check your own inbox for the email
- [ ] Verify email landed in inbox (not spam)
- [ ] Spot-check lesson link works

### Throughout Day
- [ ] Monitor Vercel analytics for traffic
- [ ] Watch Supabase for connection spikes
- [ ] Check Resend for bounce rates
- [ ] Respond to hello@curiouskelly.com

---

## If Something Breaks

### Email Not Sending
1. Check Resend dashboard
2. Verify RESEND_API_KEY in Vercel
3. Manual trigger: `curl -H "Authorization: Bearer $CRON_SECRET" https://curiouskelly.com/api/cron/daily-lesson`

### Database Errors
1. Check Supabase dashboard → Logs
2. Check connection count (max 60)
3. If needed, pause non-critical features

### Site Down
1. Check Vercel status page
2. Check Supabase status page
3. Check Cloudflare/DNS if applicable
4. Redeploy if needed: `npx vercel --prod`

### High Traffic
- Vercel auto-scales, just watch
- Database pooling handles connections
- CDN caching reduces load

---

## Emergency Contacts

- **Vercel Status**: status.vercel.com
- **Supabase Status**: status.supabase.com
- **Resend Status**: status.resend.com
- **Your Email**: hello@curiouskelly.com

---

## Success Metrics (Day 1)

- [ ] Daily email sent successfully
- [ ] No critical errors in logs
- [ ] Site stayed up 100%
- [ ] At least 1 signup from launch
- [ ] Unsubscribe rate < 5%

---

## Post-Launch (Dec 18+)

- [ ] Review Day 1 analytics
- [ ] Check email open rates in Resend
- [ ] Plan first week content/social
- [ ] Set up weekly metrics review


