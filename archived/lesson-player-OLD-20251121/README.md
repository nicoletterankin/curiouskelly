# The Daily Lesson - Lesson Player

Web-based lesson player for Kelly's Universal Classroom. Demonstrates age-adaptive learning where one lesson adapts automatically for ages 2-102.

## Features

✅ **Age Slider (2-102)** - Universal content adaptation  
✅ **Audio Playback** - Kelly's voice generated with ElevenLabs for each age variant  
✅ **Teaching Moments** - Timestamp-based learning highlights  
✅ **Interactive Choices** - Student engagement through questions and responses  
✅ **PhaseDNA Structure** - Welcome → Teaching → Practice → Wisdom  

## Quick Start

### 1. Open in Browser

```
# Using VS Code Live Server (recommended)
# Right-click on index.html → "Open with Live Server"

# OR using Python
python -m http.server 8000
# Navigate to http://localhost:8000

# OR just double-click index.html (no audio playback)
```

### 2. Test Age Adaptation

1. **Move the age slider** (2-102)
2. Watch content change:
   - Ages 2-5: "Pretty Leaves!" - Simple vocabulary
   - Ages 18-35: "The Biochemistry of Autumn" - Complex concepts  
   - Ages 61-102: "The Wisdom of Seasonal Cycles" - Reflective
3. **Click play** to hear Kelly's voice for that age
4. **Answer questions** to progress through the lesson

### 3. Audio Generation

To generate new audio files for the lesson:

```bash
cd lesson-player
python generate_audio.py
```

This will:
- Read lesson DNA from `../lessons/leaves-change-color.json`
- Call ElevenLabs API for each of the 6 age variants
- Generate MP3 files in `videos/audio/`
- Save metadata in `videos/audio/metadata.json`

**Requirements:**
- ElevenLabs API key configured
- Python 3.x
- `requests` library installed

## File Structure

```
lesson-player/
├── index.html          # Main HTML structure
├── script.js           # Lesson player logic
├── styles.css          # Styling
├── generate_audio.py   # Audio generation script
├── README.md           # This file
└── videos/
    └── audio/
        ├── kelly_leaves_2-5.mp3
        ├── kelly_leaves_6-12.mp3
        ├── kelly_leaves_13-17.mp3
        ├── kelly_leaves_18-35.mp3
        ├── kelly_leaves_36-60.mp3
        ├── kelly_leaves_61-102.mp3
        └── metadata.json
```

## How It Works

### 1. Age Adaptation

When the age slider moves, the system:
- Updates current age bucket (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- Loads appropriate lesson variant from JSON
- Updates title, description, objectives
- Loads corresponding audio file
- Adjusts vocabulary and complexity

### 2. Audio Playback

Audio files are automatically loaded based on age:
```javascript
const audioPath = `videos/audio/kelly_leaves_${ageBucket}.mp3`;
this.audioElement.src = audioPath;
this.audioElement.load();
```

### 3. Teaching Moments

Teaching moments are triggered by audio timestamps:
```javascript
// Check every second if teaching moment should appear
setInterval(() => {
    const currentTime = Math.floor(audioElement.currentTime);
    // Match timestamp in teachingMoments array
}, 1000);
```

Types of teaching moments:
- **Explanation**: Key concept explanation
- **Question**: Thought-provoking question
- **Demonstration**: Visual/kinesthetic learning
- **Story**: Narrative element
- **Wisdom**: Life lesson or reflection

### 4. Interactive Choices

Student can answer questions at specific points:
- Welcome phase questions
- Teaching phase questions  
- Practice phase questions
- Wisdom phase questions

Each choice triggers Kelly's personalized response.

## Adding New Lessons

### Step 1: Create Lesson DNA JSON

Copy the structure from `../lessons/leaves-change-color.json`:

```json
{
  "id": "lesson-id",
  "title": "Universal title",
  "description": "Universal description",
  "ageVariants": {
    "2-5": { /* age-appropriate content */ },
    "6-12": { /* ... */ },
    // ... all 6 buckets
  },
  "interactions": [ /* ... */ ],
  "metadata": { /* ... */ }
}
```

### Step 2: Generate Audio

```bash
# Edit generate_audio.py to point to your lesson file
LESSON_FILE = "../lessons/your-lesson.json"

# Run generation
python generate_audio.py
```

### Step 3: Update HTML

Edit `index.html` to load your lesson:

```javascript
// In script.js, change loadTodayLesson():
async loadTodayLesson() {
    const response = await fetch('../lessons/your-lesson.json');
    this.lessonData = await response.json();
}
```

## Configuration

### ElevenLabs API

The audio generation script uses your ElevenLabs API key:

```python
# In generate_audio.py
API_KEY = "your_api_key_here"
VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
```

Get your API key from: https://elevenlabs.io/

### Age Buckets

The 6 age buckets are:
- **2-5**: Early childhood
- **6-12**: Elementary school
- **13-17**: Teen years  
- **18-35**: Young adult
- **36-60**: Adult
- **61-102**: Elder/retirement

### Lesson Duration

Currently set to 5-10 minutes. Teaching moments should be spaced throughout the audio duration.

## Troubleshooting

### Audio won't play
- Check browser console for CORS errors
- Verify audio files exist in `videos/audio/`
- Make sure you're using a web server (not file:// protocol)

### Teaching moments not appearing
- Check console logs: `📚 Teaching moment: ...`
- Verify timestamps in lesson JSON match audio duration
- Teaching moments appear within ±1 second of timestamp

### Age slider not changing content
- Check browser console for errors
- Verify `lessons/leaves-change-color.json` exists
- Check JSON structure is valid

## Next Steps

### To Complete Full Prototype:
1. ✅ Audio generation - DONE
2. ✅ Teaching moments - DONE
3. ⏳ Add Kelly avatar video (requires CC5/iClone)
4. ⏳ Integrate with Flutter app
5. ⏳ Add more lessons (30 total for one month)

### To Add Video:
1. Create Kelly avatar in Character Creator 5
2. Import into iClone
3. Generate lipsync video for each age variant
4. Place video files in `videos/` directory
5. Update lesson JSON with video filenames

## Resources

- **Lesson DNA Schema**: `../lesson-dna-schema.json`
- **Production Workflow**: `../projects/Kelly/TODO.md`
- **Kelly Avatar Guide**: `../docs/guides/KELLY_AVATAR_WORKFLOW.md`
- **Architecture**: `../digital-kelly/ARCHITECTURE.md`

## License

Part of the UI-TARS Desktop project.

