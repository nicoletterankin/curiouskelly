# Content Creation Tools

## 🛠️ **Quick Start**

### **1. Create New Lesson**

```bash
# Copy template
cp curious-kellly/content-tools/lesson-template.json \
   curious-kellly/backend/config/lessons/your-topic.json

# Edit the lesson (use your favorite editor)
# Fill in all 6 age variants following the guide
```

### **2. Validate Lesson**

```bash
node curious-kellly/content-tools/validate-lesson.js \
  curious-kellly/backend/config/lessons/your-topic.json
```

**Expected output:**
```
🔍 Validating lesson: your-topic.json

✅ Lesson is valid and high quality!
```

### **3. Preview Lesson**

```bash
# Preview for age 35
node curious-kellly/content-tools/preview-lesson.js \
  curious-kellly/backend/config/lessons/your-topic.json \
  --age 35
```

**Shows:** Full lesson content formatted for that age

### **4. Generate Audio** (Optional)

```bash
# Generate for all age groups
node curious-kellly/content-tools/generate-audio.js \
  curious-kellly/backend/config/lessons/your-topic.json

# Or just one age group
node curious-kellly/content-tools/generate-audio.js \
  curious-kellly/backend/config/lessons/your-topic.json \
  --age-group 18-35
```

**Requires:** `ELEVENLABS_API_KEY` or `OPENAI_API_KEY` in `.env`

---

## 📁 **Files**

- **lesson-template.json** - Copy this to start a new lesson
- **lesson-authoring-guide.md** - Complete writing guide (read this!)
- **validate-lesson.js** - Validates lesson against schema + quality rules
- **preview-lesson.js** - Shows how lesson looks for any age
- **generate-audio.js** - Generates TTS audio for lesson content

---

## ✅ **Workflow**

```
1. Copy template → your-topic.json
2. Write content for all 6 age groups
3. Validate → fix any errors
4. Preview → check it looks good
5. Generate audio (optional)
6. Test in backend
7. Deploy!
```

**Time per lesson:** ~12-15 hours (with 6 age variants)  
**Goal:** 2 lessons/day = 30 lessons in 15 days

---

## 📚 **30-Lesson Curriculum**

**Week 1: Nature & Science (7 lessons)**
1. ✅ Leaves (complete)
2. ✅ Water (example complete)
3. Clouds
4. Light
5. Sound
6. Seeds
7. Stars

**Week 2: Social & Emotional (7 lessons)**
8. Friendship
9. Kindness
10. Listening
11. Patience
12. Gratitude
13. Courage
14. Curiosity

**Week 3: Physical & Movement (7 lessons)**
15. Balance
16. Breathing
17. Movement
18. Rest
19. Energy
20. Senses
21. Growth

**Week 4: Creative & Cognitive (7 lessons)**
22. Colors
23. Patterns
24. Stories
25. Music
26. Questions
27. Imagination
28. Memory

**Week 5: Connection (2 lessons + buffer)**
29. Time
30. Change

---

## 🎯 **Quality Standards**

Every lesson must:
- ✅ Work for ages 2-102 (universal)
- ✅ Have 6 complete age variants
- ✅ Pass schema validation
- ✅ Meet word count guidelines
- ✅ Include interaction prompts
- ✅ Have memorable wisdom moment
- ✅ Be safe and age-appropriate

---

## 💡 **Tips**

### **Writing Efficiency**
1. Start with ages 18-35 (baseline)
2. Simplify for younger ages (2-5, 6-12, 13-17)
3. Add depth for older ages (36-60, 61-102)
4. Use voice recorder to capture Kelly's voice
5. Batch similar ages together

### **Common Mistakes**
- ❌ Topic too narrow (e.g., "mortgage planning")
- ❌ Same language for all ages
- ❌ Forgetting to update Kelly's age
- ❌ Lessons too long
- ❌ Not testing with real people

### **Testing**
Test each lesson with someone from that age group:
- 2-5: Toddlers (parents observe)
- 6-12: Kids (ask follow-up questions)
- 13-17: Teens (get honest feedback!)
- 18-35: Adults (practical relevance?)
- 36-60: Middle-age (life experience?)
- 61-102: Elders (wisdom resonance?)

---

## 🚀 **Get Started**

1. **Read the guide:**
   ```bash
   cat curious-kellly/content-tools/lesson-authoring-guide.md
   ```

2. **Study the examples:**
   - `leaves-change-color.json` (complete)
   - `water-cycle.json` (complete)

3. **Create your first lesson:**
   ```bash
   cp lesson-template.json ../backend/config/lessons/clouds.json
   # Edit clouds.json
   node validate-lesson.js ../backend/config/lessons/clouds.json
   node preview-lesson.js ../backend/config/lessons/clouds.json --age 8
   ```

4. **Repeat for 28 more topics!**

---

## 📞 **Need Help?**

- **Validation errors?** Check `lesson-dna-schema.json` for schema
- **Writing stuck?** Review `lesson-authoring-guide.md` for each age group
- **Audio not generating?** Verify API keys in `.env`

**Goal:** 30 amazing universal lessons in 2-3 weeks! 🌍

**You've got this!** 🎉


















