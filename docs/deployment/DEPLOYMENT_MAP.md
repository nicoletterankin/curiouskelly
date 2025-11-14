# CuriousKelly.com - Deployment Map Explained

**A simple guide to understanding how Vercel, Cloudflare, and curiouskelly.com work together**

---

## 🗺️ The Big Picture

Think of your deployment like a **restaurant with two kitchens**:

```
┌─────────────────────────────────────────────────────────────┐
│                    curiouskelly.com                         │
│              (Your Restaurant - The Domain)                 │
│                                                             │
│  When someone visits curiouskelly.com, they get:           │
│  • Marketing pages (homepage, about, pricing)              │
│  • Lesson player (the actual learning app)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Kitchen 1  │ │   Kitchen 2  │ │   Delivery   │
│              │ │              │ │              │
│   Vercel     │ │  Cloudflare  │ │   GitHub     │
│  (Marketing) │ │   (Lessons)  │ │  (Updates)   │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🎯 The Three Main Players

### 1. **Cloudflare** = The Domain Manager & Lesson Host
- **Owns:** `curiouskelly.com` (you registered it through Cloudflare)
- **Manages:** DNS (where traffic goes), SSL certificates (security)
- **Hosts:** The lesson player app
- **Think of it as:** The landlord who owns the building and rents space to your lesson player

### 2. **Vercel** = The Marketing Site Host
- **Hosts:** Your marketing website (homepage, about page, pricing)
- **Builds:** Your Astro marketing site automatically
- **Think of it as:** A separate kitchen that makes the marketing pages

### 3. **GitHub** = The Source of Truth
- **Stores:** All your code
- **Triggers:** Automatic deployments when you push code
- **Think of it as:** The recipe book that both kitchens read from

---

## 🔄 How It All Works Together

### Step-by-Step Flow:

```
1. You write code
   ↓
2. You push to GitHub (git push)
   ↓
3. GitHub Actions detect the change
   ↓
4a. If marketing site changed → Deploy to Vercel
4b. If lesson player changed → Deploy to Cloudflare Pages
   ↓
5. Both services build and deploy
   ↓
6. Users visit curiouskelly.com
   ↓
7. Cloudflare DNS routes them:
   • Root (/) → Vercel (marketing)
   • /lesson-player → Cloudflare Pages (lessons)
```

---

## 🌐 Domain Routing Explained

When someone visits `curiouskelly.com`, here's what happens:

### Scenario 1: User visits homepage
```
User types: curiouskelly.com
         ↓
Cloudflare DNS checks: "Where should this go?"
         ↓
Routes to: Vercel (marketing site)
         ↓
User sees: Homepage with hero, features, signup form
```

### Scenario 2: User visits lesson player
```
User types: curiouskelly.com/lesson-player
         ↓
Cloudflare DNS checks: "Where should this go?"
         ↓
Routes to: Cloudflare Pages (lesson player)
         ↓
User sees: Interactive lesson with Kelly
```

---

## 📁 What Gets Deployed Where

### Vercel (Marketing Site)
**Location in repo:** `curiouskelly-marketing-site/` or `daily-lesson-marketing/`

**What it serves:**
- `curiouskelly.com/` → Homepage
- `curiouskelly.com/adults` → Adults page
- `curiouskelly.com/children` → Children page
- `curiouskelly.com/pricing` → Pricing page
- `curiouskelly.com/privacy` → Privacy policy

**How it deploys:**
1. GitHub Actions workflow runs (`.github/workflows/deploy-vercel.yml`)
2. Builds the Astro site (`npm run build`)
3. Pushes to Vercel
4. Vercel serves it at `curiouskelly.com`

### Cloudflare Pages (Lesson Player)
**Location in repo:** `lesson-player/`

**What it serves:**
- `curiouskelly.com/lesson-player` → The actual learning app

**How it deploys:**
1. GitHub Actions workflow runs (`.github/workflows/deploy-cloudflare.yml`)
2. Builds/prepares static files
3. Pushes to Cloudflare Pages
4. Cloudflare serves it at `curiouskelly.com/lesson-player`

---

## 🔧 The Technical Details (Simplified)

### DNS Configuration (Cloudflare)
Cloudflare manages your domain's DNS records. Think of DNS as a phone book:

```
curiouskelly.com → Points to Vercel's servers (for marketing)
/lesson-player → Points to Cloudflare Pages (for lessons)
```

### SSL Certificates
Both Vercel and Cloudflare automatically provide SSL certificates (the padlock icon). This is handled automatically - you don't need to do anything.

### Build Process

**Vercel:**
```
Marketing site code → npm run build → dist/ folder → Deploy to Vercel
```

**Cloudflare:**
```
Lesson player code → (no build needed, it's static) → Deploy to Cloudflare Pages
```

---

## 🚨 Common Confusion Points

### ❓ "Why two separate services?"

**Answer:** Because they serve different purposes:
- **Vercel** is optimized for marketing sites (Astro, fast builds, great for SEO)
- **Cloudflare Pages** is optimized for static apps (like your lesson player)
- They can both use the same domain (`curiouskelly.com`)

### ❓ "Who owns the domain?"

**Answer:** Cloudflare owns/manages the domain registration. But both Vercel and Cloudflare Pages can serve content on it.

### ❓ "How do they share the same domain?"

**Answer:** Through DNS routing:
- Cloudflare DNS says: "Root path (/) goes to Vercel"
- Cloudflare DNS says: "/lesson-player path goes to Cloudflare Pages"
- Both services are configured to accept `curiouskelly.com` as their domain

### ❓ "What happens when I push code?"

**Answer:** 
1. Code goes to GitHub
2. GitHub Actions automatically detects which files changed
3. If `curiouskelly-marketing-site/` changed → Deploys to Vercel
4. If `lesson-player/` changed → Deploys to Cloudflare Pages
5. Both deployments happen automatically (if configured correctly)

---

## 🔐 Secrets & Configuration

### GitHub Secrets (Required for auto-deployment)
These are stored in GitHub → Settings → Secrets → Actions:

**For Vercel:**
- `VERCEL_TOKEN` - API key to deploy to Vercel
- `VERCEL_ORG_ID` - Your Vercel team ID
- `VERCEL_PROJECT_ID` - Your Vercel project ID

**For Cloudflare:**
- `CLOUDFLARE_API_TOKEN` - API key to deploy to Cloudflare Pages
- `CLOUDFLARE_ACCOUNT_ID` - Your Cloudflare account ID
- `CLOUDFLARE_PROJECT_NAME` - Name of your Pages project

### Environment Variables
Each service needs its own environment variables:

**Vercel:**
- `PUBLIC_SITE_URL` = `https://curiouskelly.com`
- `ELEVENLABS_API_KEY` = (your key)
- `STRIPE_SECRET_KEY` = (your key)

**Cloudflare Pages:**
- `PUBLIC_SITE_URL` = `https://curiouskelly.com`
- `ELEVENLABS_API_KEY` = (your key)
- (other keys as needed)

---

## 🎬 The Deployment Dance

### When You Push Marketing Site Changes:

```
1. You edit: curiouskelly-marketing-site/src/pages/index.astro
2. You commit: git commit -m "Update homepage"
3. You push: git push origin main
4. GitHub Actions sees: "curiouskelly-marketing-site/** changed"
5. Triggers: deploy-vercel.yml workflow
6. Builds: npm run build in marketing site folder
7. Deploys: To Vercel
8. Result: New homepage live at curiouskelly.com
```

### When You Push Lesson Player Changes:

```
1. You edit: lesson-player/index.html
2. You commit: git commit -m "Fix lesson player"
3. You push: git push origin main
4. GitHub Actions sees: "lesson-player/** changed"
5. Triggers: deploy-cloudflare.yml workflow
6. Builds: (if needed) or just uploads files
7. Deploys: To Cloudflare Pages
8. Result: Updated lesson player at curiouskelly.com/lesson-player
```

---

## 🏗️ Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    curiouskelly.com                         │
│                  (Cloudflare DNS)                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Route: / (root)                                    │   │
│  │  → Vercel                                           │   │
│  │  → Marketing Site (Astro)                           │   │
│  │  → Pages: /, /adults, /children, /pricing          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Route: /lesson-player                              │   │
│  │  → Cloudflare Pages                                 │   │
│  │  → Lesson Player (Static HTML/JS)                   │   │
│  │  → Interactive learning app                         │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   GitHub     │ │   Vercel     │ │  Cloudflare  │
│              │ │              │ │              │
│  • Code repo │ │  • Marketing │ │  • DNS       │
│  • Workflows │ │  • Builds    │ │  • Pages     │
│  • Secrets   │ │  • Deploys   │ │  • SSL       │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## 🎯 Key Takeaways

1. **One Domain, Two Services**
   - `curiouskelly.com` is managed by Cloudflare
   - Vercel serves the marketing site
   - Cloudflare Pages serves the lesson player

2. **Automatic Deployments**
   - Push code to GitHub → Automatic deployment
   - GitHub Actions watches for changes
   - Deploys to the right service automatically

3. **Separate but Connected**
   - Marketing site and lesson player are separate apps
   - They share the same domain
   - Users navigate between them seamlessly

4. **Configuration is Key**
   - GitHub Secrets enable auto-deployment
   - Environment variables configure each service
   - DNS routing connects everything

---

## 🚀 What You Need to Set Up

### Already Done ✅
- Domain registered (curiouskelly.com)
- Cloudflare account active
- Code repository structure

### Still Needed ⏳
1. **GitHub Repository**
   - Create `nicoletterankin/curiouskelly` on GitHub
   - Push your code
   - Add GitHub Secrets

2. **Vercel Project**
   - Create project in Vercel dashboard
   - Connect to GitHub repo
   - Configure build settings
   - Add custom domain

3. **Cloudflare Pages Project**
   - Create Pages project
   - Connect to GitHub repo
   - Configure build settings
   - Add custom domain

4. **DNS Configuration**
   - Point root domain to Vercel
   - Configure /lesson-player route to Cloudflare Pages

---

## 📚 Next Steps

Once you understand this map, we can:
1. Check your current configuration
2. Identify what's missing
3. Fix any deployment issues
4. Get everything working together

**Ready to troubleshoot?** Share where you're stuck and we'll fix it step by step!

---

**Last Updated:** 2025-01-11  
**Status:** Reference Guide  
**Purpose:** Help understand deployment architecture before troubleshooting

