# Deployment Runbook

## Pre-Deployment Checklist

- [ ] All tests passing (`npm run test && npm run test:e2e`)
- [ ] Linting passes (`npm run lint`)
- [ ] Build succeeds (`npm run build`)
- [ ] Environment variables configured
- [ ] Turnstile/reCAPTCHA keys set
- [ ] CRM webhook URL configured
- [ ] Analytics IDs configured (if applicable)

## Deployment Steps

### Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Login: `vercel login`
3. Link project: `vercel link`
4. Set environment variables:
   ```bash
   vercel env add TURNSTILE_SITE_KEY production
   vercel env add TURNSTILE_SECRET_KEY production
   vercel env add CRM_WEBHOOK_URL production
   # ... etc
   ```
5. Deploy: `vercel --prod`

**Or** use GitHub integration:
- Connect repo in Vercel dashboard
- Set environment variables in UI
- Automatic deployments on push to main

### Netlify

1. Install Netlify CLI: `npm i -g netlify-cli`
2. Login: `netlify login`
3. Initialize: `netlify init`
4. Set environment variables in Netlify dashboard
5. Deploy: `netlify deploy --prod`

**Or** use GitHub integration:
- Connect repo in Netlify dashboard
- Set build command: `npm run build`
- Set publish directory: `dist`
- Set environment variables

### Cloudflare Pages

1. Install Wrangler CLI: `npm i -g wrangler`
2. Login: `wrangler login`
3. Create project in Cloudflare dashboard
4. Set environment variables in dashboard
5. Deploy: `wrangler pages deploy dist --project-name=daily-lesson-marketing`

**Or** use Git integration:
- Connect repo in Cloudflare dashboard
- Set build command: `npm run build`
- Set build output: `dist`
- Set environment variables

## Post-Deployment Verification

1. **Homepage loads**: Check `/`
2. **Form submission**: Submit test lead
3. **Thank-you page**: Verify redirect works
4. **i18n**: Check `/es-es/` and `/pt-br/`
5. **Consent manager**: Verify banner appears
6. **Analytics**: Check if tags load after consent
7. **Performance**: Run Lighthouse audit
8. **Mobile**: Test on real device

## Environment Variables Reference

### Required

- `TURNSTILE_SITE_KEY` - Cloudflare Turnstile site key
- `TURNSTILE_SECRET_KEY` - Cloudflare Turnstile secret key
- `CRM_WEBHOOK_URL` - CRM webhook endpoint

### Optional

- `GTM_ID` - Google Tag Manager ID
- `GA4_ID` - Google Analytics 4 ID
- `META_PIXEL_ID` - Meta Pixel ID
- `PUBLIC_SITE_URL` - Site URL for SEO
- `OFFER_END_DATE` - Countdown end date (ISO format)
- `ENABLE_RUM` - Enable RUM endpoint (default: false)

## Rollback Procedure

### Vercel

```bash
vercel rollback [deployment-url]
```

Or via dashboard: Deployments → Select deployment → Rollback

### Netlify

Via dashboard: Deploys → Select previous deploy → Publish deploy

### Cloudflare Pages

Via dashboard: Deployments → Select previous deployment → Retry deployment

## Monitoring

### Health Checks

- Form submission endpoint: `/api/lead`
- Homepage: `/`
- Sitemap: `/sitemap.xml`

### Error Tracking

- Check serverless function logs
- Monitor CRM webhook responses
- Review browser console errors

### Performance Monitoring

- Lighthouse CI in GitHub Actions
- Real User Monitoring (if enabled)
- Core Web Vitals tracking

## Troubleshooting

### Form Submissions Failing

1. Check Turnstile keys are correct
2. Verify CRM webhook URL is accessible
3. Check CORS settings on CRM endpoint
4. Review serverless function logs

### Analytics Not Loading

1. Verify consent manager working
2. Check GTM/GA4 IDs are set
3. Ensure marketing consent granted
4. Check browser console for errors

### Build Failures

1. Check Node.js version (18+)
2. Verify all dependencies installed
3. Review build logs for errors
4. Test locally: `npm run build`

### Deployment Errors

1. Verify environment variables set
2. Check platform-specific limits
3. Review deployment logs
4. Ensure build output directory correct

## Maintenance

### Regular Tasks

- **Weekly**: Review form submissions
- **Monthly**: Update dependencies
- **Quarterly**: Performance audit
- **Annually**: Security review

### Dependency Updates

```bash
npm audit
npm update
npm run test
npm run build
```

### Content Updates

- Edit pages in `src/pages/`
- Update translations in `src/lib/i18n/`
- Rebuild and deploy

## Emergency Contacts

- **Technical Issues**: dev@thedailylesson.com
- **Privacy Concerns**: privacy@thedailylesson.com
- **CRM Issues**: Contact CRM provider support










