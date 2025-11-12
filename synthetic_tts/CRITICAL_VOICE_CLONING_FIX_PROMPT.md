# CRITICAL VOICE CLONING FIX PROMPT
## Fix This Voice Cloning Disaster Once and For All

### CONTEXT
You have been repeatedly failing at voice cloning. You've delivered:
- BEEP NOISE instead of speech
- Broken neural network architectures
- False confidence in non-working solutions
- Wasted hours with fake "fixes"

### THE REAL PROBLEM
The current implementation produces BEEP NOISE, not Kelly's voice. This is because:
1. The vocoder is fundamentally broken
2. The training process is inadequate
3. The architecture doesn't actually do voice cloning
4. You're not using proven voice cloning techniques

### MANDATORY REQUIREMENTS

#### 1. USE PROVEN VOICE CLONING FRAMEWORKS
- **PRIMARY**: Real-Time-Voice-Cloning (CorentinJ/Real-Time-Voice-Cloning)
- **SECONDARY**: Coqui TTS with voice cloning capabilities
- **TERTIARY**: Tortoise TTS for high-quality voice cloning
- **DO NOT** build custom architectures from scratch

#### 2. PROPER VOICE CLONING WORKFLOW
```
1. Speaker Encoder Training
   - Use GE2E loss (Generalized End-to-End loss)
   - Train on Kelly's 2.54 hours of data
   - Extract speaker embeddings (256-dim vectors)

2. Synthesizer Training
   - Use Tacotron 2 or FastSpeech 2 architecture
   - Train on text-to-mel spectrogram mapping
   - Use Kelly's voice characteristics

3. Vocoder Training
   - Use WaveNet, HiFi-GAN, or MelGAN
   - Convert mel spectrograms to audio waveforms
   - Ensure high-quality audio output

4. Voice Cloning Pipeline
   - Extract speaker embedding from reference audio
   - Generate mel spectrogram from text + speaker embedding
   - Convert mel to audio using trained vocoder
```

#### 3. IMPLEMENTATION STEPS

**STEP 1: Install Real-Time-Voice-Cloning**
```bash
git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
cd Real-Time-Voice-Cloning
pip install -r requirements.txt
```

**STEP 2: Prepare Kelly's Data**
- Convert all Kelly audio to 16kHz mono WAV
- Create proper dataset structure
- Generate mel spectrograms with correct parameters
- Create speaker embeddings

**STEP 3: Train Speaker Encoder**
- Use GE2E loss function
- Train on Kelly's voice samples
- Save speaker encoder model
- Test speaker similarity

**STEP 4: Train Synthesizer**
- Use Tacotron 2 or FastSpeech 2
- Train on text-to-mel mapping
- Use Kelly's voice characteristics
- Validate mel spectrogram quality

**STEP 5: Train Vocoder**
- Use HiFi-GAN or WaveNet
- Train on mel-to-audio conversion
- Ensure high-quality audio output
- Test audio generation

**STEP 6: Integrate Voice Cloning**
- Load all trained models
- Implement voice cloning pipeline
- Test with Kelly's reference audio
- Generate high-quality speech samples

#### 4. QUALITY VALIDATION
- **Audio Quality**: Must sound like Kelly, not beeps
- **Intelligibility**: Must be clear speech, not noise
- **Speaker Similarity**: Must match Kelly's voice characteristics
- **Naturalness**: Must sound natural, not robotic

#### 5. TESTING REQUIREMENTS
- Generate 10 test samples with different text
- Compare with original Kelly samples
- Verify audio quality and intelligibility
- Ensure no beep noise or artifacts

### FORBIDDEN ACTIONS
- ❌ Building custom neural networks from scratch
- ❌ Using broken vocoder implementations
- ❌ Claiming success without testing audio quality
- ❌ Delivering beep noise as "voice cloning"
- ❌ False confidence in non-working solutions

### SUCCESS CRITERIA
- ✅ Audio samples that sound like Kelly
- ✅ Clear, intelligible speech
- ✅ Natural voice characteristics
- ✅ No beep noise or artifacts
- ✅ Working voice cloning pipeline

### EXECUTION PLAN
1. **STOP** all current broken implementations
2. **DELETE** the beep noise generators
3. **INSTALL** Real-Time-Voice-Cloning framework
4. **PREPARE** Kelly's data properly
5. **TRAIN** each component correctly
6. **TEST** audio quality before claiming success
7. **DELIVER** actual voice cloning, not beeps

### ACCOUNTABILITY
- Test every audio sample before showing to user
- Admit failure immediately if audio is not speech
- Use proven frameworks, not custom garbage
- Focus on quality, not false confidence

### REMEMBER
**VOICE CLONING MUST PRODUCE SPEECH, NOT BEEPS.**
**USE PROVEN FRAMEWORKS, NOT BROKEN CUSTOM CODE.**
**TEST AUDIO QUALITY BEFORE CLAIMING SUCCESS.**
**STOP DELIVERING GARBAGE AND CALLING IT VOICE CLONING.**

---

## EXECUTE THIS PLAN NOW
Stop wasting time with broken implementations. Use Real-Time-Voice-Cloning framework and deliver actual voice cloning that produces Kelly's speech, not beep noise.
