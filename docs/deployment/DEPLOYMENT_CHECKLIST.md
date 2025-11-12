# CuriousKelly.com - Deployment Checklist

**Quick Reference:** Use this checklist to verify all deployment components are configured correctly.

---

## ✅ Pre-Deployment

### GitHub Repository
- [ ] Repository `nicoletterankin/curiouskelly` created on GitHub
- [ ] Local repository initialized and connected
- [ ] Initial commit pushed to `main` branch
- [ ] Branch protection rules configured
- [ ] GitHub Actions enabled

### GitHub Secrets
- [ ] `VERCEL_TOKEN` - Vercel API token
- [ ] `VERCEL_ORG_ID` - Vercel team ID (Lotd)
- [ ] `VERCEL_PROJECT_ID` - Vercel project ID
- [ ] `CLOUDFLARE_API_TOKEN` - Cloudflare API token
- [ ] `CLOUDFLARE_ACCOUNT_ID` - `47ebb2a1adc311cb106acc89720e352c`
- [ ] `CLOUDFLARE_ZONE_ID` - `510107ca53356bab42f8a8d1b2de1e59`
- [ ] `CLOUDFLARE_PROJECT_NAME` - `curiouskelly`
- [ ] `APPLE_KEY_ID` - App Store Connect API key ID
- [ ] `APPLE_ISSUER_ID` - App Store Connect issuer ID
- [ ] `APPLE_KEY_CONTENT` - App Store Connect API key (base64)
- [ ] `GOOGLE_SERVICE_ACCOUNT_JSON` - Google Play service account JSON
- [ ] `PUBLIC_SITE_URL` - `https://curiouskelly.com` (optional, has default)

---

## ✅ Cloudflare Configuration

### Domain Setup
- [ ] Domain `curiouskelly.com` active in Cloudflare
- [ ] DNS records configured correctly
- [ ] SSL/TLS mode: Full (strict)
- [ ] SSL certificate issued and valid

### Pages Project
- [ ] Pages project `curiouskelly` created
- [ ] Connected to GitHub repository `nicoletterankin/curiouskelly`
- [ ] Production branch: `main`
- [ ] Build settings configured
- [ ] Custom domain `curiouskelly.com` added
- [ ] Environment variables set (see below)

### Security Features
- [ ] Page Shield: **ENABLED**
- [ ] Bot Fight Mode: **ENABLED**
- [ ] Leaked credentials detection: **ACTIVATED**

### R2 Storage
- [ ] R2 bucket `curiouskelly-assets` created
- [ ] CORS policy configured
- [ ] API token created for R2 access

### Cloudflare Environment Variables
- [ ] `ELEVENLABS_API_KEY`
- [ ] `STRIPE_SECRET_KEY`
- [ ] `ANALYTICS_ID`
- [ ] `PUBLIC_SITE_URL` = `https://curiouskelly.com`

---

## ✅ Vercel Configuration

### Project Setup
- [ ] Vercel project created for marketing site
- [ ] Connected to GitHub repository
- [ ] Team: `Lotd`
- [ ] Framework: Astro
- [ ] Build command: `npm run build`
- [ ] Output directory: `dist`
- [ ] Root directory: `curiouskelly-marketing-site` (or as configured)

### Domain Configuration
- [ ] Custom domain `curiouskelly.com` added
- [ ] DNS records configured per Vercel instructions
- [ ] SSL certificate issued

### Vercel Environment Variables
- [ ] `PUBLIC_SITE_URL` = `https://curiouskelly.com`
- [ ] `PUBLIC_DEFAULT_LOCALE` = `en-US`
- [ ] `PUBLIC_AVAILABLE_LOCALES` = `en-US,es-ES,pt-BR`
- [ ] `TURNSTILE_SITE_KEY`
- [ ] `TURNSTILE_SECRET_KEY`
- [ ] `CRM_WEBHOOK_URL`
- [ ] `CRM_AUTH_TOKEN` (optional)
- [ ] `ELEVENLABS_API_KEY`
- [ ] `STRIPE_SECRET_KEY`
- [ ] `ANALYTICS_ID`

**Note:** Set for Production, Preview, and Development environments.

---

## ✅ Flutter CI/CD

### GitHub Actions Workflow
- [ ] `.github/workflows/flutter-build.yml` created
- [ ] Workflow triggers configured
- [ ] iOS build job configured
- [ ] Android build job configured
- [ ] Artifact uploads configured

### App Store Connect (iOS)
- [ ] API key created in App Store Connect
- [ ] Key ID noted
- [ ] Issuer ID noted
- [ ] `.p8` key file downloaded
- [ ] Key base64 encoded and added to GitHub Secrets
- [ ] App created in App Store Connect (if applicable)

### Google Play Console (Android)
- [ ] Service account created in Google Cloud
- [ ] Service account linked to Play Console
- [ ] JSON key file downloaded
- [ ] JSON content added to GitHub Secrets
- [ ] App created in Play Console (if applicable)

---

## ✅ Deployment Verification

### GitHub Workflows
- [ ] Vercel deployment workflow runs successfully
- [ ] Cloudflare Pages deployment workflow runs successfully
- [ ] Flutter build workflow runs successfully
- [ ] All workflows pass on `main` branch push

### Cloudflare Pages
- [ ] Latest deployment successful
- [ ] Site accessible at `https://curiouskelly.com`
- [ ] SSL certificate valid
- [ ] No build errors in deployment logs

### Vercel
- [ ] Latest deployment successful
- [ ] Site accessible at `https://curiouskelly.com` (or preview URL)
- [ ] SSL certificate valid
- [ ] No build errors in deployment logs
- [ ] Environment variables loaded correctly

### Flutter Builds
- [ ] iOS build completes successfully
- [ ] Android APK build completes successfully
- [ ] Android AAB build completes successfully
- [ ] Artifacts downloadable from GitHub Actions

---

## ✅ Post-Deployment Testing

### Marketing Site
- [ ] Homepage loads correctly
- [ ] All pages accessible
- [ ] Forms submit successfully
- [ ] Analytics tracking working
- [ ] Multi-language support working
- [ ] Mobile responsive design verified

### Lesson Player
- [ ] Lesson player loads
- [ ] Audio playback works
- [ ] Avatar rendering works
- [ ] User interactions functional

### API Endpoints
- [ ] `/api/lead` endpoint responds
- [ ] `/api/rum` endpoint responds (if enabled)
- [ ] Error handling works correctly
- [ ] Rate limiting configured

### Mobile Apps
- [ ] iOS app builds without errors
- [ ] Android app builds without errors
- [ ] App can be installed on test devices
- [ ] Core functionality works in test builds

---

## ✅ Monitoring & Alerts

### Cloudflare
- [ ] Web Analytics enabled
- [ ] Page Shield monitoring active
- [ ] Bot analytics visible
- [ ] Error logs accessible

### Vercel
- [ ] Web Vitals tracking enabled
- [ ] Speed Insights enabled
- [ ] Error logs accessible
- [ ] Deployment notifications configured

### Application Monitoring
- [ ] Firebase Analytics configured (Flutter)
- [ ] Firebase Crashlytics configured (Flutter)
- [ ] Error tracking set up
- [ ] Performance monitoring active

---

## ✅ Documentation

- [ ] `DEPLOYMENT_ARCHITECTURE.md` reviewed
- [ ] `SETUP_GUIDE.md` followed
- [ ] `DEPLOYMENT_CHECKLIST.md` (this file) completed
- [ ] Team members have access to all services
- [ ] Credentials stored securely (password manager)
- [ ] Runbook created for common operations

---

## 🔄 Ongoing Maintenance

### Weekly
- [ ] Review deployment logs
- [ ] Check for failed deployments
- [ ] Monitor error rates
- [ ] Review security alerts

### Monthly
- [ ] Update dependencies
- [ ] Review and rotate API keys
- [ ] Audit environment variables
- [ ] Review and optimize build times
- [ ] Check storage usage (R2, etc.)

### Quarterly
- [ ] Review and update deployment documentation
- [ ] Audit security configurations
- [ ] Review and optimize costs
- [ ] Update CI/CD workflows as needed

---

## 🚨 Emergency Contacts & Resources

### Service Dashboards
- **GitHub:** https://github.com/nicoletterankin/curiouskelly
- **Cloudflare:** https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c
- **Vercel:** https://vercel.com/lotd
- **App Store Connect:** https://appstoreconnect.apple.com
- **Google Play Console:** https://play.google.com/console

### Documentation
- **GitHub Actions:** https://docs.github.com/actions
- **Cloudflare Pages:** https://developers.cloudflare.com/pages
- **Vercel Docs:** https://vercel.com/docs
- **Flutter Deployment:** https://docs.flutter.dev/deployment

### Support
- **GitHub Support:** https://support.github.com
- **Cloudflare Support:** https://support.cloudflare.com
- **Vercel Support:** https://vercel.com/support

---

## 📝 Notes

**Last Updated:** 2025-01-11  
**Completed By:** _________________  
**Next Review Date:** _________________

---

**Status Legend:**
- ✅ Complete
- ⏳ In Progress
- ❌ Blocked
- ⏸️ Paused
- 📋 Not Started

