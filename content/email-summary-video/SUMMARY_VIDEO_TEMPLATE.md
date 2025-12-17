# Daily Summary Video Template

## Purpose
A 90-120 second video for email subscribers who won't visit the app.
Delivers the core value of BOTH tracks in a single, watchable package.

---

## Video Structure (Template)

### 1. INTRO (10 seconds)
```
Good morning! I'm Kelly, and this is your daily lesson.
Today we're exploring two powerful ideas together.
```

### 2. LEARN TRACK (50-60 seconds)
```
[Opening Hook - 10s]
{learn_hook}

[Core Insight - 30s]
{learn_fact1}
{learn_fact2_condensed}

[Wisdom - 10s]
{learn_wisdom_condensed}
```

### 3. TRANSITION (5 seconds)
```
Now let's take what we learned and put it into practice.
```

### 4. GROW TRACK (30-40 seconds)
```
[Today's Skill - 10s]
Today's growth challenge: {grow_title}

[The Activity - 15s]
{grow_activity}

[Why It Matters - 10s]
{grow_why}
```

### 5. CLOSE (10 seconds)
```
That's today's lesson. Learn something, grow a little.
I'll see you tomorrow with something new.
Stay curious! ✨
```

---

## Total Duration Target: 90-120 seconds

---

## Variable Placeholders

| Variable | Source | Description |
|----------|--------|-------------|
| `{learn_hook}` | phases.hook.script | Opening hook from Learn track |
| `{learn_fact1}` | phases.fact1.script | First fact (condensed if needed) |
| `{learn_fact2_condensed}` | phases.fact2.script | Second fact (key sentence only) |
| `{learn_wisdom_condensed}` | phases.wisdom.script | Wisdom (first 2 sentences) |
| `{grow_title}` | growTrack.title | Grow track title |
| `{grow_activity}` | growTrack.activity | The action step |
| `{grow_why}` | growTrack.learning_objective | Why this matters |

---

## Email Integration

Since most email clients don't support video playback:

1. **Thumbnail + Play Button** - Static image with play icon
2. **Animated GIF** - 3-5 second preview loop
3. **Link to video** - Hosted on CDN or YouTube/Vimeo

### Recommended Approach
```html
<a href="https://curiouskelly.com/day/{day}?video=summary">
  <img src="https://curiouskelly.com/video-thumbnails/day-{day}-summary.png" 
       alt="Watch today's lesson" 
       style="border-radius: 12px;">
</a>
```

---

## HeyGen Generation

Use multi-scene video with these motion patterns:
- Intro: Motion A (warm welcoming)
- Learn content: Motion B (teaching)
- Transition: Motion C (grounded pause)
- Grow content: Motion A (warm, encouraging)
- Close: Motion A (warm close)
