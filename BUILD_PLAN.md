# Final Prototype Build Plan

## Goal
Complete a working prototype that demonstrates Kelly teaching "Why Do Leaves Change Color?" with age-adaptive audio across all 6 age buckets (2-102).

## Current State
- ✅ Web lesson player with UI and placeholders
- ✅ Lesson DNA with all 6 age variants
- ✅ ElevenLabs API integration ready
- ⚠️ No audio files generated yet
- ⚠️ No video files (can skip for initial prototype)

## Execution Plan

### Phase 1: Generate Audio Files (2-3 hours)

**Task 1.1:** Create audio generation script
- File: `lesson-player/generate_audio.py`
- Purpose: Generate 6 audio files (one per age bucket) using ElevenLabs
- Input: Script text from each age variant in `leaves-change-color.json`
- Output: `videos/audio/` directory with 6 MP3 files
- Implementation:
  - Use existing ElevenLabs voice ID: `wAdymQH5YucAkXwmrdL0`
  - Extract script text from lesson JSON for each age bucket
  - Call ElevenLabs API to generate speech
  - Save as `kelly_leaves_2-5.mp3`, `kelly_leaves_6-12.mp3`, etc.

**Task 1.2:** Run the audio generation
- Execute: `python lesson-player/generate_audio.py`
- Verify: 6 audio files created successfully

**Task 1.3:** Add audio metadata
- Create `videos/audio/metadata.json` with mapping of age bucket to audio file
- Include duration info for each audio file

### Phase 2: Update Lesson Player for Audio (1-2 hours)

**Task 2.1:** Add audio playback capability
- File: `lesson-player/script.js`
- Add HTML5 audio element alongside video element
- Add `currentAudio` property to track audio element
- Modify `loadAgeAppropriateContent()` to:
  - Load corresponding audio file for current age bucket
  - Play audio when lesson starts
  - Sync audio with lesson progress

**Task 2.2:** Create audio player UI
- File: `lesson-player/index.html`
- Add audio player controls (play/pause, progress, volume)
- Style to match video controls
- Position in video container area

**Task 2.3:** Add audio loading states
- Show loading indicator while audio loads
- Handle audio load errors gracefully
- Add retry mechanism

### Phase 3: Enhance Teaching Moments (1 hour)

**Task 3.1:** Add teaching moment triggers
- File: `lesson-player/script.js`
- Listen for audio timeupdate events
- Check if current time matches teaching moment timestamp
- Display teaching moment content when triggered
- Add visual indicators for "teaching moment happening"

**Task 3.2:** Add teaching moment UI
- Create floating panel that appears during teaching moments
- Show teaching moment type (explanation, question, story, wisdom)
- Display content in styled box
- Auto-dismiss after 3-5 seconds

**Task 3.3:** Log teaching moments
- Console log when each teaching moment triggers
- Track which moments have been shown
- Add "Teaching Moments Timeline" sidebar section

### Phase 4: Improve Age Adaptation (30 min)

**Task 4.1:** Add age-appropriate visual cues
- File: `lesson-player/styles.css`
- Change color scheme slightly based on age bucket
- Add subtle age indicator (e.g., "Age: Elementary")
- Adjust font size for very young or very old ages

**Task 4.2:** Add vocabulary highlighting
- When Kelly's script mentions key terms, highlight them
- Show vocabulary definitions on hover/click
- Differentiate simple/moderate/complex terms visually

### Phase 5: Testing & Validation (1 hour)

**Task 5.1:** Test all 6 age variants
- Move age slider through each bucket
- Verify audio plays for correct age
- Check vocabulary complexity changes
- Verify objectives update
- Test interaction choices work

**Task 5.2:** Test teaching moments
- Let audio play through
- Verify teaching moments trigger at correct times
- Check visual indicators appear
- Test teaching moment dismissal

**Task 5.3:** Test interactions
- Answer all choice questions
- Verify Kelly's responses appear
- Check lesson phases progress correctly
- Validate "wisdom" phase at end

**Task 5.4:** Browser compatibility
- Test in Chrome
- Test in Firefox
- Test in Safari (if available)
- Test responsive design on mobile

### Phase 6: Documentation (30 min)

**Task 6.1:** Update README
- File: `lesson-player/README.md`
- Document how to generate audio files
- Explain how to add new lessons
- List age adaptation features
- Include testing instructions

**Task 6.2:** Add inline code comments
- Document key functions in `script.js`
- Explain audio loading logic
- Comment on teaching moment trigger system
- Document age bucket logic

## Files to Create/Modify

### New Files:
1. `lesson-player/generate_audio.py` - Audio generation script
2. `lesson-player/videos/audio/` - Directory for audio files
3. `lesson-player/videos/audio/metadata.json` - Audio metadata
4. `lesson-player/README.md` - Documentation

### Modified Files:
1. `lesson-player/script.js` - Add audio playback logic
2. `lesson-player/index.html` - Add audio player UI
3. `lesson-player/styles.css` - Add teaching moment styling

## Success Criteria

✅ Prototype is considered complete when:
- Audio plays for each age bucket when slider moves
- Teaching moments trigger at correct timestamps
- Vocabulary complexity visibly changes with age
- Interactive choices work and show Kelly's responses
- All 6 age variants demonstrate appropriate content
- Teaching moments display correctly
- Code is commented and documented

## Estimated Time: 6-8 hours

## Ready to Execute?

Once you confirm this plan, I will:
1. Create the audio generation script
2. Generate audio files for all 6 age variants
3. Update the lesson player to handle audio
4. Add teaching moment triggers
5. Enhance UI for age adaptation
6. Test and validate the prototype

