# Deploy Landing Page in 30 Minutes
**Get curiouskelly.com live TODAY**

---

## WHAT YOU HAVE

✅ **File ready:** `curiouskelly-landing-page.html`  
✅ **Content ready:** Copy, images, structure complete  
⏳ **Need:** Domain + hosting setup

---

## FASTEST PATH: VERCEL (15 MINUTES)

### Step 1: Create Vercel Account (2 min)
1. Go to: https://vercel.com
2. Click: "Sign Up"
3. Choose: "Continue with GitHub" (easiest)
4. Allow GitHub access

### Step 2: Upload Your Site (5 min)

**Option A: Drag and Drop (EASIEST)**
1. Click: "Add New..." → "Project"
2. Scroll down to: "Or, upload your project"
3. Drag folder containing:
   - `curiouskelly-landing-page.html`
   - `public/` folder (with all assets)
4. Click: "Upload"
5. Vercel builds automatically
6. You get: `https://your-project.vercel.app`

**Option B: GitHub (Better for updates)**
1. Create GitHub repo: "curious-kelly-landing"
2. Push files:
   ```bash
   git init
   git add .
   git commit -m "Landing page"
   git remote add origin https://github.com/yourusername/curious-kelly-landing
   git push -u origin main
   ```
3. Vercel → "Import Project" → Select your repo
4. Click: "Deploy"

### Step 3: Connect Custom Domain (8 min)
1. In Vercel project → Settings → Domains
2. Add domain: `curiouskelly.com`
3. Vercel gives you DNS records:
   ```
   Type: A, Name: @, Value: 76.76.21.21
   Type: CNAME, Name: www, Value: cname.vercel-dns.com
   ```
4. Go to your domain registrar (Namecheap/GoDaddy/Cloudflare)
5. Add those DNS records
6. Wait 5-15 minutes
7. Vercel auto-configures SSL certificate
8. Visit: https://curiouskelly.com

✅ **DONE! Landing page is LIVE!**

---

## OPTION 2: CLOUDFLARE PAGES (FREE, 20 MIN)

### Step 1: Create Cloudflare Account
1. Go to: https://cloudflare.com
2. Sign up (free)
3. Add site: curiouskelly.com
4. Follow DNS migration wizard

### Step 2: Deploy to Pages
1. Dashboard → Pages → "Create a project"
2. Choose: "Direct upload"
3. Upload folder with:
   - `index.html` (rename from curiouskelly-landing-page.html)
   - All assets in `public/`
4. Click: "Deploy site"
5. You get: `https://your-project.pages.dev`

### Step 3: Connect Domain
1. Pages project → Custom domains
2. Add: `curiouskelly.com`
3. Cloudflare auto-configures (already has your DNS)
4. Wait 2-5 minutes
5. SSL automatically provisioned

✅ **DONE!**

---

## OPTION 3: NETLIFY (20 MIN)

### Similar to Vercel:
1. https://netlify.com → Sign up
2. Drag/drop folder OR connect GitHub
3. Add custom domain
4. Update DNS records
5. Wait for SSL

✅ **DONE!**

---

## IF YOU DON'T OWN CURIOUSKELLY.COM YET

### Buy Domain (10 min):

**Namecheap (Recommended):**
1. Go to: https://namecheap.com
2. Search: "curiouskelly.com"
3. Add to cart (~$12/year)
4. Checkout
5. Go to: Dashboard → Domain List → Manage
6. Advanced DNS → Add Records (from Vercel/Cloudflare)

**Cloudflare Registrar (Cheapest):**
1. Transfer OR register at: https://cloudflare.com
2. Cost: $9-10/year (no markup)
3. Integrated with Cloudflare Pages (easier setup)

---

## FILE STRUCTURE FOR DEPLOYMENT

```
curious-kelly-landing/
├── index.html               (rename curiouskelly-landing-page.html)
├── public/
│   ├── assets/
│   │   ├── branding/
│   │   │   ├── curious-kelly-logo-horizontal.png
│   │   │   └── curious-kelly-profile-pic.png
│   │   └── kelly/
│   │       ├── kelly-closeup-fullscreen.png
│   │       ├── kelly-upperbody-panelopen.png
│   │       └── (other Kelly images)
│   └── calendar/
│       └── 365_day_calendar.json
└── README.md (optional)
```

**Important:** Rename `curiouskelly-landing-page.html` → `index.html`

---

## BEFORE YOU DEPLOY

### 1. Update Links in HTML

**Find and replace in `index.html`:**
```html
<!-- OLD (placeholder) -->
<a href="#">Get Started</a>

<!-- NEW (actual links) -->
<a href="https://buy.stripe.com/YOUR_LINK">Get Started</a>
```

**Or leave as "#" for now and update post-deployment.**

### 2. Test Locally

**Option A: Python**
```bash
cd curious-kelly-landing
python -m http.server 8000
# Open: http://localhost:8000
```

**Option B: VS Code**
1. Install: "Live Server" extension
2. Right-click `index.html` → "Open with Live Server"

**Check:**
- [ ] All images load
- [ ] All links work
- [ ] Mobile responsive
- [ ] Forms submit (or show correctly)

---

## DEPLOYMENT CHECKLIST

- [ ] Domain purchased (curiouskelly.com)
- [ ] Vercel/Cloudflare/Netlify account created
- [ ] Landing page uploaded
- [ ] Custom domain connected
- [ ] DNS records added to registrar
- [ ] SSL certificate auto-provisioned (https works)
- [ ] Test on mobile
- [ ] Test on desktop
- [ ] All images load
- [ ] Links work (or show placeholders)

---

## AFTER DEPLOYMENT

### 1. Test Everything
```
✅ https://curiouskelly.com loads
✅ https://www.curiouskelly.com redirects to non-www (or vice versa)
✅ Images display
✅ Responsive on mobile
✅ Fast load time (<2 seconds)
```

### 2. Set Up Analytics (5 min)
1. Go to: https://analytics.google.com
2. Create property: "Curious Kelly"
3. Get tracking code
4. Add to `<head>` of index.html:
   ```html
   <!-- Google Analytics -->
   <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
   <script>
     window.dataLayer = window.dataLayer || [];
     function gtag(){dataLayer.push(arguments);}
     gtag('js', new Date());
     gtag('config', 'G-XXXXXXXXXX');
   </script>
   ```
5. Redeploy
6. Test: Visit site, check Real-Time report in GA

### 3. Speed Test
1. Go to: https://pagespeed.web.dev
2. Enter: curiouskelly.com
3. Get score (aim for 90+)
4. Fix any major issues

---

## TROUBLESHOOTING

### "Domain not loading"
→ DNS takes 5-60 minutes to propagate  
→ Check DNS with: `dig curiouskelly.com`  
→ Wait 1 hour and try again

### "SSL not working (not https)"
→ Wait 10-15 minutes after DNS propagates  
→ Vercel/Cloudflare auto-provision SSL  
→ Force SSL redirect in platform settings

### "Images not loading"
→ Check file paths (case-sensitive!)  
→ Ensure `public/assets/` uploaded  
→ Check browser console for 404 errors

### "Site loads but looks broken"
→ CSS not loading (check file path)  
→ Clear browser cache  
→ Check browser console for errors

---

## COST BREAKDOWN

**Free option:**
- Domain: $10-12/year (Namecheap/Cloudflare)
- Hosting: FREE (Vercel/Netlify/Cloudflare Pages)
- SSL: FREE (auto-provisioned)
- **Total: $10-12/year**

**Paid option (unnecessary):**
- Hosting: $5-20/month (if you want dedicated)
- **Not needed for landing page**

---

## AFTER IT'S LIVE

### Tell people!
```
Landing page is live! 🚀

Check out Curious Kelly:
https://curiouskelly.com

What do you think?
```

### Update social media:
- Twitter bio: https://curiouskelly.com
- Instagram bio: curiouskelly.com (no https in IG)
- LinkedIn: https://curiouskelly.com
- All other platforms

### Monitor:
- Google Analytics: Track visitors
- Errors: Check deployment logs
- Speed: Run PageSpeed tests weekly

---

## NEXT STEPS AFTER DEPLOYMENT

1. **Set up Stripe** (so "Buy" buttons work)
2. **Connect backend API** (for gift codes)
3. **Add email capture** (collect waitlist)
4. **A/B test copy** (optimize conversion)

---

## QUICK WINS

**30 minutes from now, you'll have:**
- ✅ curiouskelly.com live and working
- ✅ SSL certificate (secure https)
- ✅ Professional landing page
- ✅ Link to share on social media
- ✅ Analytics tracking visitors
- ✅ Mobile responsive site

**That's a REAL PRODUCT people can see!**

---

## DECISION TIME

**Right now:**
- [ ] Choose platform: Vercel (easiest) or Cloudflare Pages (if DNS already there)
- [ ] Open platform, create account
- [ ] Upload landing page files
- [ ] Connect domain
- [ ] Wait 15 minutes
- [ ] Visit curiouskelly.com
- [ ] CELEBRATE! 🎉

**Timer starts NOW! ⏱️ 30 minutes!**





















