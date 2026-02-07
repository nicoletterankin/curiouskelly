# v0.app Prompt: Coalition Page — READY TO BUILD

## ASSETS ALREADY CREATED (Use These)

All assets live at `/public/coalition/`. Do NOT generate images — use what's provided.

```
/public/coalition/
├── og-coalition.png          # OG image (1200x630) - old money aesthetic
├── kelly-face.png            # Kelly's portrait for hero
├── ziggurat-coalition.mp4    # 55-sec investor video (EMBED THIS)
├── coalition-data.json       # ALL DATA - fetch and render
├── ziggurat-twilight.jpg     # Building shot (cool tones)
├── ziggurat-night.jpg        # Building at night (blue)
└── hero-ziggurat-gradient.png # Hero bg with fade
```

---

## DESIGN SYSTEM: OLD MONEY

This is NOT startup energy. This is endowment, family office, generational wealth.

### Colors
```css
--bg-primary: #0a0a0a;      /* Near black */
--bg-card: #111111;         /* Slightly lighter black */
--bg-elevated: #1a1a1a;     /* Card hovers */
--text-primary: #fafafa;    /* Warm white */
--text-secondary: #a1a1aa;  /* Muted */
--text-muted: #71717a;      /* Very muted */
--accent: #3B82F6;          /* Kelly blue - ONLY accent color */
--accent-subtle: #1e40af;   /* Darker blue for borders */
--success: #22c55e;         /* Eligible/Available badges only */
```

**NO ORANGE. NO WARM TONES. NO GRADIENTS EXCEPT SUBTLE.**

### Typography
```css
/* Headlines - refined serif or clean sans */
font-family: 'Times New Roman', Georgia, serif;
/* OR */ font-family: 'Inter', system-ui, sans-serif;

letter-spacing: 0.05em;  /* Headlines slightly tracked */
line-height: 1.7;        /* Generous, readable */
```

### Spacing Philosophy
- Generous padding (py-24, px-8 on mobile, px-16 on desktop)
- Let content breathe — this is calm conviction, not urgency
- White space IS the design

---

## PAGE STRUCTURE

### 1. HERO (Full viewport)

**Background**: Autoplay the video `/coalition/ziggurat-coalition.mp4` muted, looped
**Fallback**: `/coalition/hero-ziggurat-gradient.png`

```jsx
<section className="relative h-screen flex items-center justify-center">
  <video 
    autoPlay muted loop playsInline
    className="absolute inset-0 w-full h-full object-cover opacity-40"
    src="/coalition/ziggurat-coalition.mp4"
  />
  <div className="relative z-10 text-center">
    <p className="text-sm tracking-[0.3em] text-zinc-500 mb-4">
      LESSON OF THE DAY, PBC
    </p>
    <h1 className="text-6xl md:text-8xl font-serif text-white mb-6">
      Coalition Model
    </h1>
    <p className="text-2xl text-[#3B82F6] mb-8">
      Victory of the People
    </p>
    <p className="text-zinc-400 text-sm">
      January 29, 2026 · First look for coalition partners
    </p>
    <p className="text-zinc-500 text-xs mt-4">
      Third auction in 3 days
    </p>
  </div>
</section>
```

---

### 2. THE MISSION

```jsx
<section className="py-32 px-8 text-center">
  <p className="text-xs tracking-[0.4em] text-zinc-500 mb-8">THE MISSION</p>
  <h2 className="text-4xl md:text-5xl text-white mb-8 leading-tight">
    Kelly teaches 8 billion humans.
  </h2>
  <p className="text-xl text-zinc-400 mb-4">
    Every person. Every day. For life.
  </p>
  <p className="text-[#3B82F6] text-lg">
    Loving saving grace for curious minds.
  </p>
  <div className="mt-16 text-zinc-500 text-sm space-y-2">
    <p>Alexa became the voice of the home.</p>
    <p>Siri became the voice of your device.</p>
    <p className="text-white">Kelly becomes the voice of learning.</p>
  </div>
</section>
```

---

### 3. THE ZIGGURAT

**Fetch data from `/coalition/coalition-data.json`**

Stats in elegant grid:
```jsx
<div className="grid grid-cols-2 md:grid-cols-4 gap-8">
  {/* Each stat */}
  <div className="text-center">
    <p className="text-5xl text-white font-light">1,003,041</p>
    <p className="text-xs tracking-[0.3em] text-zinc-500 mt-2">SQUARE FEET</p>
  </div>
  {/* 7 FLOORS, 92 ACRES, 1971 BUILT */}
</div>
```

Building timeline as minimal vertical line:
```
2015 ——— National Register eligible
2023 ——— First auction — no bids
2024 ——— Second auction — $177M bid cancelled
2025 ——— IRS vacates, building now empty
2026 ——— Third auction — February 1
         Our goal — Close by February 7
```

---

### 4. COALITION PARTNERS

Table from `coalition-data.json`:

```jsx
<table className="w-full">
  <thead>
    <tr className="text-left text-xs tracking-wider text-zinc-500 border-b border-zinc-800">
      <th className="py-4">PARTNER</th>
      <th className="py-4 text-right">CONTRIBUTION</th>
      <th className="py-4 text-center">TYPE</th>
      <th className="py-4 text-right">STATUS</th>
    </tr>
  </thead>
  <tbody>
    {data.coalition.map(partner => (
      <tr className="border-b border-zinc-900">
        <td className="py-4 text-white">{partner.partner}</td>
        <td className="py-4 text-right text-white">${(partner.amount/1000000).toFixed(0)}M</td>
        <td className="py-4 text-center text-zinc-400">{partner.type}</td>
        <td className="py-4 text-right">
          <span className={statusClass}>{partner.status}</span>
        </td>
      </tr>
    ))}
  </tbody>
</table>
```

Status badges:
- "Invited" = `border border-[#3B82F6] text-[#3B82F6] px-3 py-1 text-xs`
- "Eligible"/"Available" = `border border-emerald-600 text-emerald-500 px-3 py-1 text-xs`
- "To Apply" = `text-zinc-500 text-xs`

**Total: $290,000,000** (large, blue)

---

### 5. CAPITAL REQUIREMENTS

```
PHASE 1 CAPITAL

Acquisition                    $200,000,000
Closing & Due Diligence          $4,000,000
Renovation (Kelly HQ)           $75,000,000
Operating Reserve               $30,000,000
────────────────────────────────────────────
Total                          $309,000,000

Coalition Invited              $290,000,000
Gap                             $19,000,000
```

Note: "The gap represents an opportunity for additional partners."

---

### 6. FIVE-YEAR FINANCIALS

Table from `data.financials`:

| | 2026 | 2027 | 2028 | 2029 | 2030 |
|---|---|---|---|---|---|
| Revenue | $1.7M | $7.8M | $33M | $90M | $202M |
| EBITDA | ($650K) | ($2.4M) | ($3M) | $15M | $56M |
| Learners | 50K | 500K | 5M | 25M | 100M |

- Negative EBITDA in `text-zinc-500`
- Positive EBITDA in `text-[#3B82F6]`
- Learner row highlighted

---

### 7. THE FOUR PILLARS

Minimal cards:

```jsx
<div className="grid grid-cols-1 md:grid-cols-4 gap-6">
  {data.pillars.map(pillar => (
    <div className="border border-zinc-800 p-8 hover:border-zinc-700 transition">
      <h3 className="text-white text-lg mb-2">{pillar.name.toUpperCase()}</h3>
      <p className="text-zinc-400 text-sm mb-4">{pillar.description}</p>
      <p className="text-zinc-500 text-xs mb-4">{pillar.reach}</p>
      <p className="text-[#3B82F6] text-2xl">
        ${(pillar.revenue2030/1000000).toFixed(0)}M
      </p>
      <p className="text-zinc-600 text-xs">2030 Revenue</p>
    </div>
  ))}
</div>
```

---

### 8. PATH TO 8 BILLION

Vertical timeline from `data.milestones`:

```jsx
<div className="space-y-12">
  {data.milestones.map((m, i) => (
    <div className="flex gap-8">
      <div className="text-right w-24">
        <p className="text-white text-xl">{m.year}</p>
      </div>
      <div className="w-px bg-zinc-800 relative">
        <div className="absolute top-1 -left-1 w-2 h-2 rounded-full bg-[#3B82F6]" />
      </div>
      <div>
        <p className="text-3xl text-white">
          {m.learners.toLocaleString()}
        </p>
        <p className="text-zinc-400">{m.label}</p>
      </div>
    </div>
  ))}
</div>
```

---

### 9. GOVERNANCE

```jsx
<div className="grid md:grid-cols-2 gap-16">
  <div>
    <h3 className="text-xs tracking-[0.3em] text-zinc-500 mb-6">STRUCTURE</h3>
    <dl className="space-y-4">
      <div>
        <dt className="text-zinc-500 text-sm">Entity</dt>
        <dd className="text-white">Lesson of the Day, PBC</dd>
        <dd className="text-zinc-400 text-sm">California Public Benefit Corporation</dd>
      </div>
      <div>
        <dt className="text-zinc-500 text-sm">Ownership</dt>
        <dd className="text-white">Nicolette Rankin — 100%</dd>
      </div>
      <div>
        <dt className="text-zinc-500 text-sm">Operations</dt>
        <dd className="text-white">Dallas — COO</dd>
        <dd className="text-zinc-400 text-sm">Salary only, no equity</dd>
      </div>
    </dl>
  </div>
  <div>
    <h3 className="text-xs tracking-[0.3em] text-zinc-500 mb-6">COMMITMENT</h3>
    <div className="space-y-4 text-zinc-400">
      <p>This company will never be sold.</p>
      <p>This company will never go public.</p>
      <p className="text-white">This company exists to teach.</p>
    </div>
  </div>
</div>
```

---

### 10. VICTORY OF THE PEOPLE

Full-width philosophical section:

```jsx
<section className="py-32 px-8 max-w-3xl mx-auto text-center">
  <h2 className="text-3xl text-white mb-12">Victory of the People</h2>
  <div className="space-y-6 text-zinc-400 text-lg leading-relaxed">
    <p>Kelly doesn't belong to investors.<br/>
       Kelly belongs to the 8 billion people she teaches.</p>
    <p>The Ziggurat doesn't exist to generate returns.<br/>
       It exists for a hundred years as education's home.</p>
    <p className="text-white">Like libraries. Like public schools. Like the internet itself.<br/>
       Kelly becomes the foundation everyone builds upon.</p>
    <p className="text-[#3B82F6]">Not an exit. Not a multiple.<br/>
       A world where every human has a great teacher.</p>
  </div>
</section>
```

---

### 11. KEY DATES

Timeline from `data.timeline` — same style as Path to 8 Billion

---

### 12. CONTACT

```jsx
<section className="py-24 px-8 text-center">
  <h2 className="text-xs tracking-[0.3em] text-zinc-500 mb-8">CONTACT</h2>
  <p className="text-2xl text-white mb-2">Nicolette Rankin</p>
  <p className="text-zinc-400 mb-4">Founder & CEO</p>
  <a href="mailto:nicolette@thedailylesson.com" 
     className="text-[#3B82F6] hover:underline">
    nicolette@thedailylesson.com
  </a>
  
  <div className="mt-16 max-w-xl mx-auto text-left">
    <p className="text-zinc-500 text-sm mb-6">Questions to consider:</p>
    <ul className="space-y-3 text-zinc-400 text-sm">
      <li>Does the $200M acquisition price feel right?</li>
      <li>What would you need to say yes?</li>
      <li>Who else belongs at this table?</li>
      <li>What governance would give you confidence?</li>
    </ul>
  </div>
</section>
```

---

### 13. FOOTER

```jsx
<footer className="py-16 px-8 border-t border-zinc-900 text-center">
  <p className="text-xs tracking-[0.3em] text-zinc-500 mb-4">
    LESSON OF THE DAY, PBC
  </p>
  <p className="text-zinc-600 text-sm mb-8">
    California Public Benefit Corporation · Founded 2022
  </p>
  <div className="space-x-8 text-sm">
    <a href="https://thedailylesson.com" className="text-zinc-400 hover:text-white">
      thedailylesson.com
    </a>
    <a href="https://curiouskelly.com" className="text-zinc-400 hover:text-white">
      curiouskelly.com
    </a>
  </div>
  <div className="mt-12 text-zinc-600 text-sm">
    <p>Kelly teaches.</p>
    <p>The Ziggurat stands.</p>
    <p>The people learn.</p>
  </div>
</footer>
```

---

## META TAGS

```html
<title>Coalition Model — Lesson of the Day, PBC</title>
<meta name="description" content="Kelly teaches 8 billion humans. Join the coalition building education infrastructure for humanity." />
<meta property="og:title" content="Coalition Model — Victory of the People" />
<meta property="og:description" content="An invitation to build Kelly's home. $309M coalition for the Ziggurat." />
<meta property="og:image" content="/coalition/og-coalition.png" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
```

---

## INTERACTIONS

1. **Smooth scroll** — Sticky nav with section anchors (minimal, just "Mission | Coalition | Timeline | Contact")
2. **Video background** — Muted autoplay loop in hero
3. **Hover states** — Subtle: `hover:bg-zinc-900` on cards, `hover:border-zinc-700` on borders
4. **Mobile** — Stack all grids, reduce padding, keep elegance
5. **NO countdown timers, NO flashing, NO urgency indicators**

---

## DATA FETCHING

```typescript
// Fetch all data from single JSON
const data = await fetch('/coalition/coalition-data.json').then(r => r.json())

// Contains:
// data.meta (dates)
// data.ziggurat (building stats)
// data.coalition (partners array)
// data.financials (5-year by year)
// data.pillars (4 pillars)
// data.milestones (path to 8B)
// data.timeline (key dates)
// data.ownership (governance)
// data.contact (Nicolette)
```

---

## SUMMARY

Build a single-page Next.js application that is:

- **Old Money** — Black, white, Kelly blue (#3B82F6). NO warm tones.
- **Minimal** — White space is the design
- **Trustworthy** — This is a $309M ask to mission-aligned partners
- **Calm** — Conviction without urgency
- **Complete** — Everything a partner needs to understand and respond

The reader finishes thinking:
*"This is infrastructure. This is permanent. I want to be part of this."*

---

## URL

`thedailylesson.com/coalition`
