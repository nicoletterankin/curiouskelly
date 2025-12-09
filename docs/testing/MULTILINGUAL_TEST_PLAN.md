# 🌍 Multilingual Test Plan for Curious Kelly

**Version:** 1.0.0  
**Created:** December 2025  
**Status:** Active

---

## 📋 Overview

This document outlines the test cases for verifying multilingual support in the Curious Kelly lesson player. All tests ensure learners in EU territories (Spanish, French) receive proper localized content.

---

## 🎯 Test Objectives

1. Verify language selection persists across sessions
2. Confirm text content displays in selected language
3. Validate voice synthesis uses correct language model
4. Ensure fallback to English when translations missing
5. Test all 6 age variants × 3 languages

---

## ✅ Manual Test Cases

### Test 1: Language Selector UI

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Open `public/learn.html` | Page loads with default language (EN) |
| 2 | Click language button (🌐) in right rail | Language popover/expand opens |
| 3 | Select "Español" | Badge updates to "ES", popover closes |
| 4 | Verify `localStorage.getItem('kelly_language')` | Returns `"es"` |
| 5 | Refresh page | Language badge still shows "ES" |

### Test 2: Spanish Content Display (Day 1)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Set language to ES | Badge shows "ES" |
| 2 | Navigate to `?day=1` | Lesson loads |
| 3 | Check Hook phase text | "¡Hola pequeño amigo! 🌟 Hoy vamos a aprender algo súper genial: ¡El Sol!" (for 2-5 age) |
| 4 | Check Wisdom phase text | Contains "Nuestra estrella da vida a todo en la Tierra" |

### Test 3: French Content Display (Day 1)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Set language to FR | Badge shows "FR" |
| 2 | Navigate to `?day=1` | Lesson loads |
| 3 | Check Hook phase text | "Salut petit ami ! 🌟 Aujourd'hui, nous allons apprendre quelque chose de super cool : Le Soleil !" |
| 4 | Check Wisdom phase text | Contains "Notre étoile donne la vie à tout sur Terre" |

### Test 4: Voice Synthesis (Spanish)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Set language to ES | Badge shows "ES" |
| 2 | Load Day 1, unmute audio | Audio plays |
| 3 | Listen to Kelly's voice | Voice speaks Spanish text with Spanish pronunciation |
| 4 | Check browser Network tab | `/api/tts` called with Spanish text |
| 5 | Verify model used | `eleven_multilingual_v2` (supports all languages) |

### Test 5: Voice Synthesis (French)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Set language to FR | Badge shows "FR" |
| 2 | Load Day 1, unmute audio | Audio plays |
| 3 | Listen to Kelly's voice | Voice speaks French text with French pronunciation |

### Test 6: Age Variant × Language Matrix

Test each combination displays correctly:

| Age Bucket | EN | ES | FR |
|------------|----|----|-----|
| 2-5 | ✓ Playful tone | ✓ "pequeño amigo" | ✓ "petit ami" |
| 6-12 | ✓ Curious tone | ✓ Age-appropriate ES | ✓ Age-appropriate FR |
| 13-17 | ✓ Direct tone | ✓ Age-appropriate ES | ✓ Age-appropriate FR |
| 18-35 | ✓ Conversational | ✓ Age-appropriate ES | ✓ Age-appropriate FR |
| 36-60 | ✓ Measured tone | ✓ Age-appropriate ES | ✓ Age-appropriate FR |
| 61-102 | ✓ Warm tone | ✓ Age-appropriate ES | ✓ Age-appropriate FR |

### Test 7: Fallback Behavior

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Manually edit DNA file to remove `es` key | Simulate missing translation |
| 2 | Set language to ES | Badge shows "ES" |
| 3 | Load affected day | Falls back to English text |
| 4 | Console shows | Warning about missing translation (no crash) |

### Test 8: Days 11-30 Topic Translations

Verify new topics have proper translations:

| Day | EN Topic | ES Topic | FR Topic |
|-----|----------|----------|----------|
| 11 | Why We Yawn | Por qué bostezamos | Pourquoi nous bâillons |
| 15 | How Computers Think | Cómo piensan las computadoras | Comment pensent les ordinateurs |
| 20 | Electricity | La Electricidad | L'Électricité |
| 25 | Dinosaurs | Los Dinosaurios | Les Dinosaures |
| 30 | Why We Age | Por qué envejecemos | Pourquoi nous vieillissons |

---

## 🔊 Expected Audio Output

### ElevenLabs Voice Model IDs

| Language | Voice ID | Model |
|----------|----------|-------|
| English | `wAdymQH5YucAkXwmrdL0` (Kelly) | `eleven_multilingual_v2` |
| Spanish | `wAdymQH5YucAkXwmrdL0` (Kelly) | `eleven_multilingual_v2` |
| French | `wAdymQH5YucAkXwmrdL0` (Kelly) | `eleven_multilingual_v2` |

**Note:** The `eleven_multilingual_v2` model automatically detects and speaks the correct language based on the input text. Kelly's voice ID remains consistent across languages.

---

## ⚠️ Edge Cases

### Empty Translation Object
```javascript
// Bad: Missing language key
"hook": { "en": "Hello", "es": "Hola" }  // No "fr"

// Expected: Fallback to "en"
getVariantText(phase) → "Hello"
```

### Null/Undefined Content
```javascript
// Bad: Null phase
phase = null

// Expected: Return empty string, no crash
getVariantText(null) → ""
```

### Mixed Content Types
```javascript
// Bad: String instead of object
"hook": "Hello world"  // Not localized

// Expected: Return the string directly
getVariantText(phase) → "Hello world"
```

---

## 🚀 Automated Validation

### DNA Schema Validation
Run to verify all generated lessons have EN/ES/FR:

```bash
node -e "
const fs = require('fs');
const lessons = fs.readdirSync('./generated/lessons').filter(f => f.endsWith('.json') && f !== 'manifest.json');
let issues = [];
lessons.forEach(f => {
  const data = JSON.parse(fs.readFileSync('./generated/lessons/' + f));
  Object.entries(data.ageVariants).forEach(([age, variant]) => {
    Object.entries(variant.phases).forEach(([phase, content]) => {
      if (!content.en || !content.es || !content.fr) {
        issues.push(f + ' ' + age + ' ' + phase + ': missing language');
      }
    });
  });
});
console.log(issues.length ? issues.join('\n') : '✅ All lessons have EN/ES/FR content');
"
```

---

## 📊 Success Criteria

- [ ] All 365 lessons have EN/ES/FR content in all phases
- [ ] Language selector works and persists
- [ ] Voice plays in correct language
- [ ] No console errors when switching languages
- [ ] Fallback to EN works gracefully
- [ ] All 6 age variants work per language

---

## 📝 Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| QA Lead | | | Pending |
| Engineering | | | Pending |
| Product | | | Pending |

---

*Document maintained by the Global Expansion Agent*






