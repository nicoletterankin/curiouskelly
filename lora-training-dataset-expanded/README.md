# Kelly LoRA Training Dataset - Cinematic Perfection Edition

This dataset contains **25 curated reference images** for training a production-grade character LoRA.

## Dataset Composition

### Original 7 Images (from initial training)
- 4.jpeg - Close-up, big smile
- pray.jpeg - Hands together, hopeful
- open-walk.jpeg - Full body walking, profile
- square-chair2.jpeg - Seated, hand on heart
- our-girl.jpeg - Seated, chin on hand
- open.png - Close-up, contemplative
- close.jpeg - Eyes closed, peaceful

### 18 NEW Expansion Images
- three-quarter-left.png -  three-quarter view from left, seated
- three-quarter-right.png -  three-quarter view from right, seated
- front-full-body.png -  full body, standing
- surprised-delighted.png -  close-up, surprised delighted expression
- teaching-explaining.png -  medium shot, teaching explaining
- curious-questioning.png -  close-up, curious questioning expression
- encouraging-supportive.png -  medium shot, encouraging supportive expression
- celebrating-joyful.png -  full body, arms raised celebrating
- concentrating-focused.png -  close-up, concentrated focused expression
- laughing-genuine.png -  close-up, genuine laugh
- pointing-left.png -  medium shot, pointing left
- pointing-right.png -  medium shot, pointing right
- pointing-up.png -  medium shot, pointing up
- arms-crossed-confident.png -  medium shot, arms crossed
- leaning-forward-engaged.png -  medium shot, leaning forward
- standing-casual.png -  full body, standing casual
- medium-shot-neutral.png -  medium shot waist up, neutral relaxed expression
- extreme-closeup-eyes.png -  extreme close-up, face focus

## Coverage Matrix

| Category | Coverage |
|----------|----------|
| **Angles** | Front, 3/4 Left, 3/4 Right, Profile |
| **Expressions** | Smile, Thoughtful, Surprised, Teaching, Curious, Celebrating, Laughing, Focused |
| **Poses** | Seated, Standing, Walking, Pointing L/R/Up, Arms Crossed, Leaning Forward |
| **Framing** | Extreme Close-up, Close-up, Medium, Full Body |

## Training Settings (Replicate)

- **Trainer:** ostris/flux-dev-lora-trainer
- **Trigger word:** kelly
- **Steps:** 2500 (increased for larger dataset)
- **LoRA rank:** 32 (increased for more detail)
- **Learning rate:** 0.0001

## Why This Works

With 25 diverse images covering all angles, expressions, and poses:
- The LoRA learns Kelly's **identity**, not just specific poses
- New generations maintain consistency regardless of prompt
- Hands, expressions, and proportions remain stable
- Every frame is recognizably Kelly

Generated: 2025-12-27T15:37:26.757Z
Version: 2.0 - Cinematic Perfection Edition
